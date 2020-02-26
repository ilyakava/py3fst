"""Iterate through hyperparams of ST to get accuracies
"""

import os
import time

import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm

import hyper_pixelNN as hsinn

import pdb

FEAT_DIR = '/scratch0/ilya/locDoc/data/hyperspec/features/npz_feat'
FEAT_DIR = '/scratch2/ilyak/locDoc/data/hyperspec/features/npz_feat'

RESULTS_DIR = '/scratch0/ilya/locDoc/pyfst'
RESULTS_DIR = '/scratch2/ilyak/locDoc/pyfst'

def run_svm(trainX, trainY, testX, testY):
    print('starting training')
    start = time.time()
    clf = SVC(kernel='linear')
    clf.fit(trainX, trainY)
    end = time.time()
    print('Training done. Took %is' % int(end - start))

    # now test
    test_chunk_size = 1000

    n_correct = 0
    for i in tqdm(range(0,testY.shape[0],test_chunk_size), desc='Testing'):
        p_label = clf.predict(testX[i:i+test_chunk_size]);
        n_correct += (p_label == testY[i:i+test_chunk_size]).sum()
    acc = float(n_correct) / testY.shape[0]
    return acc
        

def get_acc_for_config(dataset, st_net_spec, preprocessed_data_path, preprocessed_data_root, mask_paths, save_features):
    """One set of hyperparams and dataset, several trials
    """
    trainimgname, trainlabelname = hsinn.dset_filenames_dict[dataset]
    trainimgfield, trainlabelfield = hsinn.dset_fieldnames_dict[dataset]
    data, labels = hsinn.load_data(trainimgname, trainimgfield, trainlabelname, trainlabelfield)
    
    if save_features:
        data = hsinn.load_or_preprocess_data(data, preprocessed_data_path, preprocessed_data_root, st_net_spec, 31)
    else:
        data = hsinn.preprocess_data(data, st_net_spec, 31)
    
    accs = []
    # loop through masks
    for mask_path in mask_paths:
        train_mask = hsinn.multiversion_matfile_get_field(mask_path, 'train_mask')
        val_mask = hsinn.multiversion_matfile_get_field(mask_path, 'test_mask')
        trainX, trainY, valX, valY = hsinn.get_train_val_splits(data, labels, train_mask, val_mask, (0,0,0)) # , n_eval=2048
    
        # run svm
        accs.append(run_svm(trainX.squeeze(), trainY, valX.squeeze(), valY))
    print('[%s] Avg acc is %.2f' % (hsinn.spec_to_str(st_net_spec), sum(accs) / len(accs)))
    return accs
    

def specs1():
    """
    Spatial psis should be decreasing, phi should be equal to smallest psi
    """
    specs = []
    myrange = [9,7,5,3]
    for b in myrange:
        for s1 in myrange:
            for s2 in myrange:
                if s1 >= s2:
                    specs.append( [ [b,s1,s1], [b,s2,s2], [b,s2,s2] ] )
    return specs
    
def specs_append9():
    """
    Spatial psis should be decreasing, phi should be equal to smallest psi
    """
    specs = []
    myrange = [9,7,5,3]
    for b in myrange:
        for s1 in [9]:
            for s2 in myrange:
                if s1 >= s2:
                    specs.append( [ [b,s1,s1], [b,s2,s2], [b,s2,s2] ] )
    return specs
    
def specs_append9_bands():
    """
    Spatial psis should be decreasing, phi should be equal to smallest psi
    """
    specs = []
    myrange = [7,5,3]
    for b in [9]:
        for s1 in myrange:
            for s2 in myrange:
                if s1 >= s2:
                    specs.append( [ [b,s1,s1], [b,s2,s2], [b,s2,s2] ] )
    return specs
    
def perform_gridsearch(dataset, masks, specs, outfile='gridsearch.npz', save_features=False, preprocessed_data_root=FEAT_DIR):
    # a list< list < int > > -> list< float >.
    # Maps a configuration to a set of accuracies
    results = {}
    for spec_i, spec in enumerate(specs):
        psi1,psi2,phi = spec
    
        st_net_spec = hsinn.st_net_spec_struct(psi1,psi2,phi)
    
        preprocessed_data_path = os.path.join(preprocessed_data_root, '%s__%s.npz' % (dataset, hsinn.spec_to_str(st_net_spec)))
    
        accs = get_acc_for_config(dataset, st_net_spec, preprocessed_data_path, preprocessed_data_root, masks, save_features)
        results[hsinn.spec_to_str(st_net_spec)] = accs
        print('FINISHED %i/%i' % (spec_i+1, len(specs)))
    
        np.savez(os.path.join(RESULTS_DIR, outfile), results=results)
    print('Saved %s' % os.path.join(RESULTS_DIR, outfile))
    

def paviaU():
    dataset = 'PaviaU'
    # masks = ['PaviaU_gt_traintest_s03_1_3f6384.mat',
    #          'PaviaU_gt_traintest_s03_2_b67b5f.mat',
    #          'PaviaU_gt_traintest_s03_3_7d8356.mat',
    #          'PaviaU_gt_traintest_s03_4_241266.mat',
    #         'PaviaU_gt_traintest_s03_5_ccbbb1.mat',
    #         'PaviaU_gt_traintest_s03_6_dce186.mat',
    #         'PaviaU_gt_traintest_s03_7_d5cdfe.mat',
    #         'PaviaU_gt_traintest_s03_8_6bcd5a.mat',
    #         'PaviaU_gt_traintest_s03_9_a1ff2b.mat',
    #         'PaviaU_gt_traintest_s03_10_e1dac2.mat']
    
             
    masks = ['PaviaU_strictsinglesite_trainval_s90_0_700083.mat',
'PaviaU_strictsinglesite_trainval_s90_1_509297.mat',
'PaviaU_strictsinglesite_trainval_s90_2_291093.mat',
'PaviaU_strictsinglesite_trainval_s90_3_232341.mat',
'PaviaU_strictsinglesite_trainval_s90_4_833337.mat',
'PaviaU_strictsinglesite_trainval_s90_5_672053.mat',
'PaviaU_strictsinglesite_trainval_s90_6_215589.mat',
'PaviaU_strictsinglesite_trainval_s90_7_553699.mat',
'PaviaU_strictsinglesite_trainval_s90_8_734449.mat',
'PaviaU_strictsinglesite_trainval_s90_9_484519.mat']
    
    # masks = [os.path.join('/scratch0/ilya/locDoc/data/hyperspec/',m) for m in masks]
    # perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_paviaU_s200_10trials_full_new.npz', save_features=True)
#     masks = ['PaviaU_strictsinglesite_trainval_s50_0_979087.mat',
# 'PaviaU_strictsinglesite_trainval_s50_1_270161.mat',
# 'PaviaU_strictsinglesite_trainval_s50_2_259715.mat',
# 'PaviaU_strictsinglesite_trainval_s50_3_107251.mat',
# 'PaviaU_strictsinglesite_trainval_s50_4_729473.mat',
# 'PaviaU_strictsinglesite_trainval_s50_5_129325.mat',
# 'PaviaU_strictsinglesite_trainval_s50_6_927653.mat',
# 'PaviaU_strictsinglesite_trainval_s50_7_627051.mat',
# 'PaviaU_strictsinglesite_trainval_s50_8_525881.mat',
# 'PaviaU_strictsinglesite_trainval_s50_9_785489.mat']


    masks = [os.path.join('/scratch0/ilya/locDoc/data/hyperspec/',m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_paviaU_strictsingle_site_s90_10trials.npz', save_features=True)

def Botswana():
    dataset = 'Botswana'
    masks = ['Botswana_strictsinglesite_trainval_s20_0_840407.mat',
'Botswana_strictsinglesite_trainval_s20_1_027359.mat',
'Botswana_strictsinglesite_trainval_s20_2_059593.mat',
'Botswana_strictsinglesite_trainval_s20_3_800757.mat',
'Botswana_strictsinglesite_trainval_s20_4_848729.mat',
'Botswana_strictsinglesite_trainval_s20_5_309369.mat',
'Botswana_strictsinglesite_trainval_s20_6_913005.mat',
'Botswana_strictsinglesite_trainval_s20_7_508879.mat',
'Botswana_strictsinglesite_trainval_s20_8_218687.mat',
'Botswana_strictsinglesite_trainval_s20_9_288573.mat']

    masks = [os.path.join('/cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/',m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_Bots_singlesite_s20_10trials.npz')

def KSC():
    dataset = 'KSC'
    masks = ['KSC_strictsinglesite_trainval_s50_0_665848.mat',
'KSC_strictsinglesite_trainval_s50_1_783796.mat',
'KSC_strictsinglesite_trainval_s50_2_455308.mat',
'KSC_strictsinglesite_trainval_s50_3_643924.mat',
'KSC_strictsinglesite_trainval_s50_4_514244.mat',
'KSC_strictsinglesite_trainval_s50_5_762052.mat',
'KSC_strictsinglesite_trainval_s50_6_668196.mat',
'KSC_strictsinglesite_trainval_s50_7_990120.mat',
'KSC_strictsinglesite_trainval_s50_8_191432.mat',
'KSC_strictsinglesite_trainval_s50_9_307364.mat']

    masks = [os.path.join('/scratch0/ilya/locDoc/data/hyperspec/', m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_KSC_strictsinglesite_s50_10trials.npz', save_features=True)

def IP():
    dataset = 'IP'
    masks = ['IP_strictsinglesite_trainval_s20_0_211328.mat',
'IP_strictsinglesite_trainval_s20_1_881136.mat',
'IP_strictsinglesite_trainval_s20_2_869530.mat',
'IP_strictsinglesite_trainval_s20_3_014114.mat',
'IP_strictsinglesite_trainval_s20_4_586540.mat',
'IP_strictsinglesite_trainval_s20_5_805378.mat',
'IP_strictsinglesite_trainval_s20_6_666022.mat',
'IP_strictsinglesite_trainval_s20_7_581782.mat',
'IP_strictsinglesite_trainval_s20_8_901868.mat',
'IP_strictsinglesite_trainval_s20_9_778852.mat']

    masks = [os.path.join('/cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/', m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_IP_strictsinglesite_s20_10trials.npz', save_features=True)



def main():
    # paviaU()
    Botswana()
    # IP()

if __name__ == '__main__':
    main()