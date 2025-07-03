#import classes from other files 
from gtxDLClassAWSUtils import Utils
from HelperFunc import Helper
from ModelArchitecture import ModelInit
from DataImport import Operations

class DL(Utils, Helper, ModelInit, Operations):    
    # Initialization method runs whenever an instance of the class is initiated
    def __init__(self):
        super().__init__()
        self.bucket = '20240909-hikaru'
        
        isCase = 'Default'#input('Choose a case:\n Default\n Default4fx')
        self.case = isCase
        self.params = self.Defaults(self.case) # Define some default parameters based on the selected case
        self.isTesting = False
        self.AWS = False       
        self.model_name = 'sub_3D_u_net_with_3D_at_beg_2D_at_encoder_decoder_with_thickness_info' #'original_siamese_3DCNN' #

if __name__ == "__main__":

    pass

        
    
    
    
    