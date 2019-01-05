import unittest
import numpy as np
import sys
sys.path.insert(0,'..')
import preprocessing3 as proc
import data_loader
# Testing functions

class TestPreprocessing(unittest.TestCase):
    

    def test_ema_mngu0_channels(self):
        data = proc.ema_read("../dataset2/mngu0_s1_0494.ema")
        second_dim = data.shape[1] 
        self.assertEqual(14,second_dim)

    def test_ema_mocha_channels(self):
        data = proc.ema_read_mocha("../dataset/msak0_154.ema")
        second_dim = data.shape[1] 
        self.assertEqual(14,second_dim)

    def test_chunk(self):
        fold = 5
        lennum = 15
        idx = np.arange(lennum)
        chunks = proc.k_split(idx,fold)
        all_chunks = np.concatenate(chunks, axis=0)
        print(len(chunks))
        self.assertEqual(len(chunks),fold)
        self.assertEqual(all_chunks.shape[0],lennum)
        
    def test_validation_splitter(self):

        fold = 10
        file_list = [str(i) for i in range(11)]
        train_idx, val_idx = proc.train_val_split(file_list, 0.3, k=fold)
        self.assertEqual(type(train_idx),type(list()))
        self.assertEqual(type(val_idx),type(list()))

        for i in range(fold):
            total_idx = len(train_idx[i]) + len(val_idx[i])
            self.assertEqual(total_idx, len(file_list))
            
class TestDataLoader(unittest.TestCase):
    def test_2D_delay_shape(self):
        signal = np.array([[2,2],[1,1]])
        signal_delayed = data_loader.delay_signal(signal,1)
        self.assertEqual(signal_delayed.shape,signal.shape)
    def test_3D_delay_shape(self):
        signal = np.array([[2,2],[1,1],[3,3]])
        signal_delayed = data_loader.delay_signal(signal,1)
        self.assertEqual(signal_delayed.shape,signal.shape)
    def test_data_gen(self):
        options = {}
        options["batch_size"] = 45
        options["delay"] = 1
        options["k"] = 0
        options["num_features"] = 15
        options["save_dir"] = "../processed_comb2_filtered_3"
        options["out_features"] = 82
        val_gen = data_loader.DataGenerator(options,False,True)
        something, thing = val_gen.__getitem__(0)
        train_gen = data_loader.DataGenerator(options,True,True)
if __name__ == '__main__':
    unittest.main()
    
