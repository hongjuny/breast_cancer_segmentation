from model import *
from data import *
import numpy as np

np.random.seed()


### train
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


myGene = trainGenerator( 2,'data/breast/train','image','label',data_gen_args,save_to_dir = None)
valGene = trainGenerator( 2, 'data/breast/val', 'image', 'label', data_gen_args, save_to_dir= None )

model = unet()
model_checkpoint = ModelCheckpoint('unet_segmentation.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
early_stopping = EarlyStopping( monitor= 'val_loss', patience= 5, restore_best_weights= True )

model.fit_generator(
        myGene,steps_per_epoch=2000,epochs=100,
        validation_data= valGene, validation_steps= 1000,
        callbacks=[
            model_checkpoint,
            early_stopping,
            ])


### test
testGene = testGenerator("data/breast/test")
model = unet()
model.load_weights("unet_segmentation.hdf5")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/breast/test",results)
