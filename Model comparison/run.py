import random
import os
import numpy as np
import scipy.io
from tqdm import tqdm
from newdataloaer import DataLoaderIQA
import torch
from scipy import stats
import argparse
import models
import csv


def getReuslts(len, type, results_dir):
    plccs = 0.0
    srccs = 0.0
    for t in range(len):
        d1 = scipy.io.loadmat((os.path.join(results_dir,'test_gt%s_cnt%06d.mat' % (type, t))))['gt']
        d2 = scipy.io.loadmat((os.path.join(results_dir,'test_pred%s_cnt%06d.mat' % (type, t))))['pred']

        srcc_val, _ = stats.spearmanr(d1.squeeze(), d2.squeeze())
        plcc_val, _ = stats.pearsonr(d1.squeeze(), d2.squeeze())
        if plcc_val > plccs:
            plccs = plcc_val
        if srcc_val > srccs:
            srccs = srcc_val
    return plccs, srccs


class Config(dict):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Config(value)
        return value


def DataSetup(dataset, root, batch_size, data_lens, ratio, pre_proc):
    scn_idxs = [x for x in range(data_lens)]
    np.random.shuffle(scn_idxs)
    scn_idxs_train = scn_idxs[:int(ratio * data_lens)]
    scn_idxs_test = scn_idxs[int(ratio * data_lens):]

    loader_train = DataLoaderIQA(dataset, root, scn_idxs_train, batch_size=batch_size, istrain=True, pre_proc=pre_proc).get_data()
    loader_test = DataLoaderIQA(dataset, root, scn_idxs_test, batch_size=batch_size, istrain=False, pre_proc=pre_proc).get_data()
    return loader_train, loader_test


def test_model(model, loaders, config, cnt, results_dir):
    torch.cuda.empty_cache()
    model.train(False)
    model.eval()
    my_device = torch.device('cuda:0')
    pred_vals = np.empty((0, 1))
    gt_vals = np.empty((0, 1))
    bcnt = 0
    for inputs, labels in loaders.test:
        inputs, labels = inputs.float().to(my_device), labels.float().to(my_device)
        pred = model(inputs)

        pred_vals = np.append(pred_vals, pred.detach().cpu().numpy(), axis=0)
        gt_vals = np.append(gt_vals, labels[:, None].detach().cpu().numpy(), axis=0)
        bcnt += 1

    scipy.io.savemat(os.path.join(results_dir, 'test_gt%s_cnt%06d.mat' % (config.type, cnt)), {'gt': gt_vals})
    scipy.io.savemat(os.path.join(results_dir, 'test_pred%s_cnt%06d.mat' % (config.type, cnt)), {'pred': pred_vals})

    srcc_val, _ = stats.spearmanr(pred_vals.squeeze(), gt_vals.squeeze())
    plcc_val, _ = stats.pearsonr(pred_vals.squeeze(), gt_vals.squeeze())

    model.train(True)
    return srcc_val, plcc_val


def train_model(model, loaders, optims, config, results_dir):
    torch.cuda.empty_cache()
    model.train(True)
    my_device = torch.device('cuda:0')
    best_accuracy = 0.0
    
    for t in range(config.nepoch):
        pred_vals = np.empty((0, 1))
        gt_vals = np.empty((0, 1))
        epoch_loss = []

        for inputs, labels in tqdm(loaders.train):
            inputs, labels = inputs.float().to(my_device), labels.float().to(my_device)
            pred = model(inputs)
            lossA = optims.criterion(pred.squeeze(), labels.detach().squeeze())
            loss = lossA
            optims.optimA.zero_grad()
            loss.backward()
            optims.optimA.step()
            optims.schedA.step()

            epoch_loss.append(lossA.item())
            pred_vals = np.append(pred_vals, pred.detach().cpu().numpy(), axis=0)
            gt_vals = np.append(gt_vals, labels[:, None].detach().cpu().numpy(), axis=0)

        print('testing....')
        srcc_val_t, _ = stats.spearmanr(pred_vals.squeeze(), gt_vals.squeeze())
        plcc_val_t, _ = stats.pearsonr(pred_vals.squeeze(), gt_vals.squeeze())

        srcc_val, plcc_val = test_model(model, loaders, config, t, results_dir)
        print(
            'Test Phase: %05d SRCC : %.4f\t  PLCC : %.4f\t Train Phase=> SRCC: %.4f \t PLCC %.4f\t RecLoss: %.4f\t' % (
                t, srcc_val, plcc_val, srcc_val_t, plcc_val_t, sum(epoch_loss) / len(epoch_loss)))
        # torch.save(cur_model, './results/model.pkl')

        scipy.io.savemat(os.path.join(results_dir, 'train_gt%s_cnt%06d.mat' % (config.type, t)), {'gt': gt_vals})
        scipy.io.savemat(os.path.join(results_dir, 'train_pred%s_cnt%06d.mat' % (config.type, t)), {'pred': pred_vals})

        # record data
        new_input_data = [f"{t:05d}", f"{srcc_val:.4f}", f"{plcc_val:.4f}", f"{srcc_val_t:.4f}", f"{plcc_val_t:.4f}",
                          f"{sum(epoch_loss) / len(epoch_loss):.4f}"]
        csv_file_path = os.path.join(results_dir, "Accuracy_result.csv")
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_input_data)
        # save model
        if plcc_val > best_accuracy:
            best_accuracy = plcc_val
            model_save_path = os.path.join(results_dir, 'best_model.pth')
            if config.model_name in ['Swin_avg_fine_single', 'Vit_token_fine_multi']:
                torch.save(model.state_dict(), model_save_path)
            else:
                torch.save(model, model_save_path)

            

def context_initial(cmd):
    #10073
    dataset_info = {
        'koniq': {'len': 10073,  #10073
                  'path': r'autodl-fs/koniq10k_1024x768'},
        'livec': {'len': 1162,   #1162
                  'path': r'autodl-fs/livec/LIVEC'},
        'tid': {'len': 3000,     #3000
                'path': r'autodl-fs/tid'},
        'kadid': {'len': 10125,  #10125
                  'path': r'autodl-fs/kadid/KADID'},
    }

    iqa_model = {
        'Resnet50_gap_fine_single': models.Resnet50_gap_fine_single,
        'Resnet50_gmp_fine_single': models.Resnet50_gmp_fine_single,
        'Resnet50_mix_fine_single': models.Resnet50_mix_fine_single,
        'Resnet50_sp_fine_single': models.Resnet50_sp_fine_single,
        'Resnet50_std_fine_single': models.Resnet50_std_fine_single,
        #'Resnet50_gap_end2end_single': models.Resnet50_gap_end2end_single,
        #'Resnet50_gap_fixed_single': models.Resnet50_gap_fixed_single,
        'Resnet50_gap_fine_multi': models.Resnet50_gap_fine_multi,
        'Resnet50_spp_fine_single':models.Resnet50_spp_fine_single,
        'Resnet50_spp_fine_multi':models.Resnet50_spp_fine_multi,
        'Vgg19_gap_fine_single': models.Vgg19_gap_fine_single,
        'Vgg19_gmp_fine_single': models.Vgg19_gmp_fine_single,
        'Vgg19_mix_fine_single': models.Vgg19_mix_fine_single,
        'Vgg19_sp_fine_single': models.Vgg19_sp_fine_single,
        'Vgg19_std_fine_single': models.Vgg19_std_fine_single,
        #'Vgg19_gap_end2end_single': models.Vgg19_gap_end2end_single,
        #'Vgg19_gap_fixed_single': models.Vgg19_gap_fixed_single,
        'Vgg19_spp_fine_single':models.Vgg19_spp_fine_single,
        'Vgg19_spp_fine_multi':models.Vgg19_spp_fine_multi,
        'Vgg19_gap_fine_multi': models.Vgg19_gap_fine_multi,
        'InceptionResnetV2_gap_fine_single': models.InceptionResnetV2_gap_fine_single,
        'InceptionResnetV2_gmp_fine_single': models.InceptionResnetV2_gmp_fine_single,
        'InceptionResnetV2_mix_fine_single': models.InceptionResnetV2_mix_fine_single,
        'InceptionResnetV2_sp_fine_single': models.InceptionResnetV2_sp_fine_single,
        'InceptionResnetV2_std_fine_single': models.InceptionResnetV2_std_fine_single,
        #'InceptionResnetV2_gap_end2end_single': models.InceptionResnetV2_gap_end2end_single,
        #'InceptionResnetV2_gap_fixed_single': models.InceptionResnetV2_gap_fixed_single,
        'InceptionResnetV2_spp_fine_single':models.InceptionResnetV2_spp_fine_single,
        'InceptionResnetV2_spp_fine_multi':models.InceptionResnetV2_spp_fine_multi,
        'InceptionResnetV2_gap_fine_multi': models.InceptionResnetV2_gap_fine_multi,
        'Vit_token_fine_single': models.Vit_token_fine_single,
        'Vit_avg_fine_single': models.Vit_avg_fine_single,
        'Vit_token_end2end_single': models.Vit_token_end2end_single,
        'Vit_token_fixed_single': models.Vit_token_fixed_single,
        'Vit_token_fine_multi': models.Vit_token_fine_multi,
        'Swin_avg_fine_single': models.Swin_avg_fine_single,
        'Swin_avg_end2end_single': models.Swin_avg_end2end_single,
        'Swin_avg_fixed_single': models.Swin_avg_fixed_single,
        'Swin_avg_fine_multi': models.Swin_avg_fine_multi
    }


    myconfigs = {
        'lr': cmd.lr,
        'weight_decay': 5e-4,
        'T_MAX': 50,
        'eta_min': 0,
        'nepoch': cmd.epoch,
        'batch_size': cmd.batch_size,
        'data_lens': dataset_info[cmd.dataset]['len'],
        'root': dataset_info[cmd.dataset]['path'],
        'type': '%04d' % cmd.sd,
        'pre_proc': cmd.pre_proc,
        'ratio': cmd.ratio,
        'dataset': cmd.dataset,
        'model':iqa_model[cmd.model],
        'model_name': cmd.model
    }


    cur_seed = cmd.sd
    random.seed(cur_seed)
    os.environ['PYTHONHASHSEED'] = str(cur_seed)
    np.random.seed(cur_seed)
    torch.manual_seed(cur_seed)
    torch.cuda.manual_seed_all(cur_seed)
    torch.backends.cudnn.deterministic = True

    return Config(myconfigs)


def main(cmd):

    myconfig = context_initial(cmd)
    model = myconfig.model().cuda()
    model_name = model.__class__.__name__
    print("Model Name:", model_name)

    optimizerA = torch.optim.Adam(model.parameters(), lr=myconfig.lr, weight_decay=myconfig.weight_decay)
    train_loader, test_loader = DataSetup(dataset=myconfig.dataset,
                                          root=myconfig.root,
                                          batch_size=myconfig.batch_size,
                                          data_lens=myconfig.data_lens,
                                          ratio=myconfig.ratio,
                                          pre_proc=myconfig.pre_proc)
    schedulerA = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerA, myconfig.T_MAX, myconfig.eta_min)

    criterion = torch.nn.MSELoss()

    optim_params = Config({'criterion': criterion,
                           'optimA': optimizerA,
                           'schedA': schedulerA,
                           })

    data_loaders = Config({'train': train_loader,
                           'test': test_loader})


    results_dir = os.path.join('./results', model_name + '_' + myconfig.dataset + '_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    header = ["Epoch", "Test_SRCC", "Test_PLCC", "Train_SRCC", "Train_PLCC", "RecLoss"]
    csv_file_path = os.path.join(results_dir, "Accuracy_result.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    train_model(model, data_loaders, optim_params, myconfig, results_dir)

    plcc_max, srcc_max = getReuslts(myconfig.nepoch, myconfig.type, results_dir)
    
    print("plcc_max:"+str(plcc_max)+","+"srcc_max:"+str(srcc_max))
    
    ##txt record
    #log_message = f"CurModelName: {model_name} | Seed: {myconfig.type} | MaxPLCC: {plcc_max:.4f} | MaxSRCC: {srcc_max:.4f} | Proc: {myconfig.pre_proc}"
    #with open("LOG.txt", "a") as log_file:
        #log_file.write(log_message + "\n")
        
    #csv record
    #data = [
    #    ("CurModelName", "Seed", "MaxPLCC", "MaxSRCC", "Proc"),
    #    (model_name, myconfig.type, plcc_max, srcc_max, myconfig.pre_proc)]
    #with open("LOG.csv", "a", newline='') as file:
    #    writer = csv.writer(file)
    #    if file.tell() == 0:
    #        writer.writerow(data[0])
    #    writer.writerow(data[1])
    #
    #os.system("rm -rf ./results/*")

    print('OK..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='sd', type=int, default=400, help='Random Seed')
    parser.add_argument('--proc', dest='pre_proc', type=str, default='resize1', help='Pre-Proc Type: Reise|Crop|...')
    parser.add_argument('--ratio', dest='ratio', type=float, default=0.8, help='Ratio of Train vs. Test')
    parser.add_argument('--data', dest='dataset', type=str, default='koniq', help='Involved DataSet')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning Rate')
    parser.add_argument('--bs', dest='batch_size', type=int, default=12, help='Batch Size')
    parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='train epochs')
    parser.add_argument('--model', dest='model', type=str, default='Resnet50_gap_fine_single', help='model')


    cmd = parser.parse_args()
    main(cmd)

