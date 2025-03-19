import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import utils
import models
import plots
import os

OUTDIR = '/Users/irf3irf3/Desktop/offline_workspace/data/tissue_of_origin/ML_output_Fresh'


def E2E(data_dict, tumor_fraction,MODEL_SETUP):
    OOF_real = []
    OOF_mix1 = []
    OOF_mix2 = []
    
    Train_test_summary = dict() 
    LOOCV_summary = dict()
    for feature, datas in data_dict.items():
        data = datas[0]
        train_data = data[data["split"] == "training"].drop(columns=["split"])
        test_data = data[data["split"] == "testing"].drop(columns=["split"])


        
        for key in MODEL_SETUP:
            #train test models
            print('\n\n⏳ running train/test')
            tmp_trained_model=  models.train_model(
                                            train_data, 
                                            "cohort", 
                                            MODEL_SETUP[key][0], 
                                            MODEL_SETUP[key][1], 
                                        
                                        )

            predict_folder_path = OUTDIR+'/'+feature+'_'+key
            os.makedirs(predict_folder_path, exist_ok=True)
            _, temp_auc = models.predict_model(tmp_trained_model, test_data, plot_cm=True, plot_roc=True, save_figures_path=predict_folder_path)
            
            
            Train_test_summary[feature+'_'+key+'_auc'] = temp_auc
            #trained_model_with_auc[key]= [tmp_trained_model, temp_auc]

            utils.LOD(tmp_trained_model, datas[1] , datas[2],tumor_fraction, figure_path = predict_folder_path)



            

            #LOOCV models
            print('\n\n⏳ running LOOCV')

            LOOCV_folder_path = OUTDIR+'/'+feature+'_'+key+'_LOOCV'
            os.makedirs(LOOCV_folder_path, exist_ok=True)
            temp_LOOCV_model, temp_loocv_auc, oof_df = models.train_model_loocv(train_data, test_data, "cohort", MODEL_SETUP[key][0], MODEL_SETUP[key][1], save_figures_path=LOOCV_folder_path) ##🚀 need to use full data before final run
            
            LOOCV_summary[feature+'_'+key+'_LOOCV_auc'] =   temp_loocv_auc 

            LOD_pred1, LOD_pred2 = utils.LOD(temp_LOOCV_model, datas[1] , datas[2],tumor_fraction, figure_path = LOOCV_folder_path)


            if key=='LogReg':
              
                OOF_real.append(oof_df.add_prefix(feature+"_").copy(deep=True))
                LOD_pred1['True Label'] = LOD_pred1.index.map(utils.assign_label)
                OOF_mix1.append( LOD_pred1.add_prefix(feature+"_Probability_").copy(deep=True))
                LOD_pred2['True Label'] = LOD_pred2.index.map(utils.assign_label)
                OOF_mix2.append(LOD_pred2.add_prefix(feature+"_Probability_").copy(deep=True))

                
                

    if OOF_real:
        OOF_real_df = utils.merge_and_clean_labels(OOF_real)
        OOF_mix1_df = utils.merge_and_clean_labels(OOF_mix1)
        OOF_mix2_df = utils.merge_and_clean_labels(OOF_mix2)
    
        LogRegAllThreeFeatures = dict()
        
        LogRegAllThreeFeatures['LogRegAllThreeFeatures'] = [OOF_real_df, OOF_mix1_df, OOF_mix2_df]
        
    
    
    if Train_test_summary and LOOCV_summary:
        plots.plot_auc_heatmap(Train_test_summary, save_figure_path=OUTDIR+"/Train_test_summary.png")
        plots.plot_auc_heatmap(LOOCV_summary, save_figure_path=OUTDIR+"/LOOCV_summary.png")
    
    if 'LogRegAllThreeFeatures' in locals(): 
        return LogRegAllThreeFeatures 

   


        
    




