import matplotlib.pyplot as plt
import numpy as np, h5py
import os 
import pandas as pd 
import scipy.io
import tensorflow as tf 
import dicttoxml 
from xml.dom.minidom import parseString
import matplotlib.pyplot as plt 
import matplotlib
#from sklearn import metrics
from keras.models import Model, load_model
import io
import os, time, sys
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import StandardScaler

from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout, MaxPool2D, UpSampling2D, ZeroPadding2D, Activation, SpatialDropout2D

from DataImport import Operations
import boto3
import mat73
from os.path import isfile, join
import time
import tempfile
import os
import numpy as np 
#from DataImport import MyDataset
class MonteCarloDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
        
class Helper(Operations):
    def __init__(self):
        super().__init__()

    def Plot(self,isTraining=False):
            while True:
                if isTraining==True:
                    # Plot loss curves
                    plt.plot(self.history.history['loss'])
                    plt.plot(self.history.history['val_loss'])
                    plt.title('Loss Curves')
                    plt.ylabel('Loss (MSE)')
                    plt.xlabel('Epoch #')
                    plt.legend(['Training', 'Validation'], loc='upper right')
                    plt.yscale('log')
                    plt.show() 
                    break
                elif isTraining==False:
                    log_files = []
                    for folder in os.listdir("ModelParameters"):
                        if not folder.endswith((".h5",".log",".xml")):
                            for file in os.listdir("ModelParameters/"+folder):
                                if file.endswith(".log"):
                                    if  'params' not in file:
                                        filename = "ModelParameters/"+folder+'/'+file
                                        log_files.append(filename)
                                        print(filename)
                    name = input('Input the absolute path to the log file: ')
                    history = pd.read_csv(name)
                    # Plot loss curves
                    plt.plot(history['loss'])
                    plt.plot(history['val_loss'])
                    plt.title('Loss Curves')
                    plt.ylabel('Loss (MSE)')
                    plt.xlabel('Epoch #')
                    plt.legend(['Training', 'Validation'], loc='upper right')
                    plt.yscale('log')
                    plt.show() 
                    break
                else:
                    print('You didn\'t select a valid option, type Y/N, or enter nothing to escape: ')
            return None

    

    
    def get_min(self, DF):

        DF_zeros = np.array(DF)

        DF_min_per_case = np.nanmin(DF_zeros, axis = (1,2))

        return DF_min_per_case
    
    def get_max(self, val):

        val = np.array(val)

        val_min_per_case = np.max(val, axis = (1,2))

        return val_min_per_case


    def import_data_for_testing(self):
        
        self.importData(isTesting=True,quickTest=True)

        
    def load(self):

        s3_client = boto3.client('s3')

        print("inside")

        keras_files = []
        
        time.sleep(1.5)

        #choose to import data from bucket or the local file path "ModelParameters"
        from_S3 = 0
        
        if from_S3:
            bucket = self.bucket
            folder = "ModelParameters"
            s3 = boto3.resource("s3")
            s3_bucket = s3.Bucket(bucket)
            files_in_s3 = []
            for f in s3_bucket.objects.filter(Prefix=folder).all():
                key_parts = f.key.split(folder + "/")
                if len(key_parts) > 1 and key_parts[1]:  # skip root folder or malformed keys
                    files_in_s3.append(key_parts[1])

            #filter files 
            for file in files_in_s3:
                if file.endswith((".keras")):
                    filename = "ModelParameters/"+ file 
                    keras_files.append(filename)
                    print(filename)
        else: 
            #display the model parameters available for export 
            for folder in os.listdir("ModelParameters"):
                #if not folder.endswith((".keras")):
                for file in os.listdir("ModelParameters/"+folder):
                    if file.endswith((".keras")):
                        filename = "ModelParameters/"+folder+'/'+file
                        keras_files.append(filename)
                        print(filename)  
        while True: # Loop until user makes acceptable choice for the input directory
            loadFile = input('Enter the general and specific directory pertaining to the .keras (weights) file you would like to load: ')
            if loadFile == '': # User enters nothing; break
                break
            elif loadFile in keras_files:

                if from_S3:

                    #load file from s3 
                    obj = s3_client.get_object(Bucket=self.bucket, Key=loadFile)

                    # Read the binary content of the file
                    model_data = obj['Body'].read()

                    # Create a temporary file to store the model
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
                        # Write the binary data to the temporary file
                        tmp_file.write(model_data)
                        tmp_file_path = tmp_file.name  # Get the path to the temporary file

                    # Load the model from the temporary file

                    self.modelD = load_model(tmp_file_path, compile=False)
                    
                    os.remove(tmp_file_path)  # If you want to delete the temp file manually

                    break
                else:
                    #if running from local file 
                    self.modelD = load_model(loadFile, compile=False)
                    
                    #self.modelD = self.Model_tf(model_name = self.model_name)

                    #self.modelD.load_weights(loadFile)  # or .h5
                    break

            else: # If the modelD attribute does not exist
                print('\nModel is not currently defined - select again.') 
                break

        print("end")
                
    def visualize_feature_maps(self, num_layer):

        #choose model 
        self.modelD_visualize = Model(inputs=self.modelD.inputs, outputs=self.modelD.layers[num_layer].output)

        print("loaded")
        
    def Analysis(self, save_image = 0):
        
        use_same_data = 0
        iceberg = 0
        if not use_same_data:
            self.import_data_for_testing()

        self.indxIncl = np.nonzero(self.temp_DF_pre_conversion)

        #self.Predict()
        self.OP = np.array(self.OP)
        self.FL = np.array(self.FL) #scale by 2

        # if self.thickness is not None:
        #     predict = self.modelD.predict([self.OP, self.FL, self.thickness], batch_size = 32)  
        # else:
        #     predict = self.modelD.predict([self.OP, self.FL], batch_size = 32)  
        predict = self.modelD.predict([self.OP, self.FL], batch_size = 32)  
        QF_P = predict[0]
        DF_P = predict[1]
        #mask_P = predict[2] if len(predict) > 2 else None  # Check if mask is returned
        QF_P /= self.params['scaleQF']
        DF_P /= self.params['scaleDF']  

        self.save = 'n'
    
        DF_P = np.reshape(DF_P, (DF_P.shape[0], DF_P.shape[1], DF_P.shape[2]))
        QF_P = np.reshape(QF_P, (QF_P.shape[0], QF_P.shape[1], QF_P.shape[2]))
        #mask_P = np.reshape(mask_P, (mask_P.shape[0], mask_P.shape[1], mask_P.shape[2])) if mask_P is not None else None
        ## Error Stats
        # Average error
        
        DF_error = DF_P - self.DF
        QF_error = QF_P - self.QF
        DF_erroravg = np.mean(abs(DF_error[self.indxIncl]))
        DF_errorstd = np.std(abs(DF_error[self.indxIncl]))
        QF_erroravg = np.mean(abs(QF_error[self.indxIncl]))
        QF_errorstd = np.std(abs(QF_error[self.indxIncl]))
        print('Average Depth Error (SD): {}({}) mm'.format(float('%.5g' % DF_erroravg),float('%.5g' % DF_errorstd)))
        print('Average Concentration Error (SD): {}({}) ug/mL'.format(float('%.5g' % QF_erroravg),float('%.5g' % QF_errorstd)))

        depth_true = self.DF[self.indxIncl]
        depth_error_abs = np.abs(DF_error[self.indxIncl])

        regions = [(0,5), (5,10), (10,15)]
        for (dmin, dmax) in regions:
            mask = (depth_true >= dmin) & (depth_true < dmax)
            if np.any(mask):
                avg_err = np.mean(depth_error_abs[mask])
                std_err = np.std(depth_error_abs[mask])
                print(f"Depth Error {dmin}-{dmax} mm: {avg_err:.3f} ± {std_err:.3f} mm (N={np.sum(mask)})")
            else:
                print(f"Depth range {dmin}-{dmax} mm: No data")


        # Overall  mean squared error
        DF_mse = np.sum((DF_P - self.DF) ** 2)
        DF_mse /= float(DF_P.shape[0] * DF_P.shape[1] * DF_P.shape[2])
        QF_mse = np.sum((QF_P - self.QF) ** 2)
        QF_mse /= float(QF_P.shape[0] * QF_P.shape[1] * QF_P.shape[2])
        print('Depth Mean Squared Error: {} mm'.format(float('%.5g' % DF_mse)))
        print('Concentration Mean Squared Error: {} ug/mL'.format(float('%.5g' % QF_mse)))

        # Max and Min values per sample

        if iceberg:
            DF_min = self.get_max(self.DF)
            DFP_min = self.get_max(DF_P)
        else:
            DF_min = self.get_min(self.DF)
            DFP_min = self.get_min(DF_P)
            
        DF_min = np.array(DF_min)
        DFP_min = np.array(DFP_min)
        plot_save_path =  os.path.join('./predictions/' + self.folder_name)

        if save_image:
            np.savetxt(plot_save_path + "_depth_predict_min.txt", DFP_min, fmt="%.4f")  # You can adjust fmt for formatting
            #np.savetxt(plot_save_path + "_depth_truth_min.txt", DF_min, fmt="%.4f")  # You can adjust fmt for formatting

        #compute absolute mindepth error 
        min_depth_error = np.mean(np.abs(DFP_min - DF_min))
        min_depth_error_std = np.std(np.abs(DFP_min - DF_min))
        print("Average Minimum Depth Error (SD) : {min_depth_error} ({min_depth_error_std})".format(min_depth_error = min_depth_error, min_depth_error_std = min_depth_error_std))

        print("Minimum Depth (mm): ", DFP_min)
        ## Plot Correlations
        
        fig, (plt1, plt2) = plt.subplots(1, 2)
        
        plt1.scatter(self.DF[self.indxIncl],DF_P[self.indxIncl],s=1)
        plt1.set_xlim([0, 10.5])
        plt1.set_ylim([0, 10.5])
        y_lim1 = plt1.set_ylim()
        x_lim1 = plt1.set_xlim()
        plt1.plot(x_lim1, y_lim1,color='k')
        plt1.set_ylabel("Predicted Depth (mm)")
        plt1.set_xlabel("True Depth (mm)")
        plt2.scatter(self.QF[self.indxIncl],QF_P[self.indxIncl],s=1)
        plt2.set_xlim([0, 10])
        plt2.set_ylim([0, 10])
        plt2.set_ylabel("Predicted Concentration (ug/mL)")
        plt2.set_xlabel("True Concentration (ug/mL)")
        plt.tight_layout()
        if self.save in ['Y','y']:
            plot_save_path_depth_and_concentration = plot_save_path + '_DF_QF.png'
            plt.savefig(plot_save_path_depth_and_concentration, dpi=100, bbox_inches='tight')
        plt.show()
       

        font = {'family': 'DejaVu Sans', 'weight': 'bold', 'size':20}
        matplotlib.rc('font', **font)
        min_depth_graph = plt.figure()

        #compute R^2 
        from sklearn.metrics import r2_score
        print("R2: ", r2_score(DF_min, DFP_min))
        
        plt.scatter(DF_min,DFP_min,s=3, label =  "Correct Classification", color = ['red'])

        DF_min_classify = np.array(DF_min) < 5 
        DFP_min_classify = np.array(DFP_min) < 5
        
        failed_result = DF_min_classify !=DFP_min_classify
        failed_result = np.squeeze(failed_result)
        
        #plt.scatter(DF_min[failed_result],DFP_min[failed_result],label = "Incorrect Classification", s=3, color = ['red'])
        #plt.legend(loc="upper left", prop={'size': 13, 'weight':'bold'})

   

        plt.xlim([0, 11])
        plt.ylim([0, 11])
        plt.plot(plt.xlim([0, 11]), plt.ylim([0, 11]),linestyle='dashed', linewidth=3, color = 'b')
        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("True Depth (mm)")
        #plt.title("Minimum Depth")
        plt.tight_layout()
        plt.yticks([0, 2, 4, 6, 8, 10])
        plt.xticks([0, 2, 4, 6, 8, 10])

        
        min_depth_graph.show()
        

        #define path to save the predictions 
        if save_image:
            self.save = 'y'
        plot_save_path =  os.path.join('./predictions/' + self.folder_name)

        
            
        for i in range(np.shape(self.DF)[0]):#np.shape(self.DF)[0]#range(num_plot_display):
            print("DF, DF pred: ", DF_min[i], DFP_min[i])
            print("index_num: ", i)
            fig, axs = plt.subplots(2,3)
            plt.set_cmap('jet')
            plt.colorbar(axs[0,0].imshow(self.DF[i,:,:],vmin=0,vmax=23), ax=axs[0, 0],fraction=0.046, pad=0.04, ticks = [0, 5, 10, 15])
            

            axs[0,0].axis('off')
            axs[0,0].set_title('True Depth (mm)')
        
            
            plt.colorbar(axs[0,1].imshow(DF_P[i,:,:],vmin=0,vmax=23), ax=axs[0, 1],fraction=0.046, pad=0.04, ticks = [0,5 , 10, 15])
            axs[0,1].axis('off')
            axs[0,1].set_title('Predicted Depth (mm)')
            plt.colorbar(axs[0,2].imshow(abs(DF_error[i,:,:]),vmin=0,vmax=10), ax=axs[0, 2],fraction=0.046, pad=0.04)
            axs[0,2].axis('off')
            axs[0,2].set_title('|Error (mm)|')
            plt.colorbar(axs[1,0].imshow(self.QF[i,:,:],vmin=0,vmax=10), ax=axs[1, 0],fraction=0.046, pad=0.04)
            axs[1,0].axis('off')
            axs[1,0].set_title('True Conc (ug/mL)')
            plt.colorbar(axs[1,1].imshow(QF_P[i,:,:],vmin=0,vmax=10), ax=axs[1, 1],fraction=0.046, pad=0.04)
            axs[1,1].axis('off')
            axs[1,1].set_title('Predicted Conc (ug/mL)')
            plt.colorbar(axs[1,2].imshow(abs(QF_error[i,:,:]),vmin=0,vmax=10), ax=axs[1, 2],fraction=0.046, pad=0.04)
            #axs[0,2].text(5, 5, 'min_depth error = ' + str(temp_value_str_1 - temp_value_str_2), bbox={'facecolor': 'white', 'pad': 4}, size = 'x-small')

            axs[1,2].axis('off')
            axs[1,2].set_title('|Error (ug/mL)|')
            plt.tight_layout()
                            
            if self.save in ['Y', 'y']:
                # Define base name
                print("inside")
                # base_filename = plot_save_path + f'_sample_{i}_'
                cmap = 'jet'

                base_filename = plot_save_path + f'_sample_{i}_panel.png'
    
                # Save the entire 2x3 subplot figure
                fig.savefig(base_filename, dpi=300, bbox_inches='tight')
                
                def save_with_colorbar(data, vmin, vmax, title, filename, ticks=None):
                    fig, ax = plt.subplots()
                    cax = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
                    cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
                    #cbar.ax.tick_params(labelsize=10, labelfontfamily = 'bold')  # Set tick label font size

                    if ticks:
                        cbar.set_ticks(ticks)
                    
                    cbar.ax.set_yticklabels(ticks, fontsize=30, fontweight='bold')

                    #ax.set_title(title, fontsize=16, fontweight='bold', fontname='Times New Roman')
                    ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(filename, dpi=100, bbox_inches='tight')
                    plt.close(fig)  # Close the figure to free memory

                # # Save plots with colorbars
                # save_with_colorbar(self.DF[i, :, :], 0, 15, 'True Depth (mm)', base_filename + 'true_depth.png', ticks=[0, 5, 10, 15])
                # save_with_colorbar(DF_P[i, :, :], 0, 15, 'Predicted Depth (mm)', base_filename + 'predicted_depth.png', ticks=[0, 5, 10, 15])
                # save_with_colorbar(abs(DF_error[i, :, :]), 0, 10, '|Depth Error| (mm)', base_filename + 'depth_error.png', ticks=[0, 5, 10])
                # save_with_colorbar(self.QF[i, :, :], 0, 10, 'True Conc (μg/mL)', base_filename + 'true_conc.png', ticks=[0, 5, 10])
                # save_with_colorbar(QF_P[i, :, :], 0, 10, 'Predicted Conc (μg/mL)', base_filename + 'predicted_conc.png', ticks=[0, 5, 10])
                # save_with_colorbar(abs(QF_error[i, :, :]), 0, 10, '|Conc Error| (μg/mL)', base_filename + 'conc_error.png', ticks=[0, 5, 10])
                            

            plt.show()
        
    def PrintFeatureMap(self):
        """Generate Feature Maps"""
        feature_maps = self.modelFM.predict([self.OP, self.FL]) # Output for each layer
        layer_names = [layer.name for layer in self.modelFM.layers] # Define all the layer names
        layer_outputs = [layer.output for layer in self.modelFM.layers] # Outputs of each layer
        print('Feature map names and shapes:')
        for layer_name, feature_map in zip(layer_names, feature_maps):
            print(f" {layer_name} shape is --> {feature_map.shape}")
        while True:
            fm = input('Choose the feature map you want to display: ')
            if fm in layer_names:
                self.fm = fm
                break
            else:
                print('Invalid entry, try again')       
        for layer_name, feature_map in zip(layer_names, feature_maps):  
            if layer_name == self.fm:
                if len(feature_map.shape) == 5:
                    for j in range(4): # Number of feature maps (stick to 4 at a time)
                        for i in range(self.params['nF']): # Spatial frequency
                            feature_image = feature_map[0, :, :, i, j]
                            feature_image-= feature_image.mean()
                            feature_image*=  64
                            feature_image+= 128
                            plt.figure(  )
                            plt.title ( layer_name +' Filter: '+str(j+1)+' SF: '+str(i+1) )
                            plt.grid  ( False )
                            plt.imshow( feature_image, aspect='auto')
                else:
                    for i in range(1): # Number of feature maps
                        feature_image = feature_map[0, :, :, i]
                        feature_image-= feature_image.mean()
                        feature_image*=  64
                        feature_image+= 128
                        plt.figure(  )
                        plt.title ( layer_name +' Filter: ' +str(i+1))
                        plt.grid  ( False )
                        plt.imshow( feature_image, aspect='auto')