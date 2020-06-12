from model import *
from data import *


### train
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

<<<<<<< HEAD
myGene = trainGenerator( 2,'/home/ay5/Documents/git/unet/data/membrane/train','image','label',data_gen_args,save_to_dir = None)
=======
myGene = trainGenerator( 2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
>>>>>>> 6b681003fa361416c54272bbb71ffc113a884692
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])


### test
<<<<<<< HEAD
testGene = testGenerator("/home/ay5/Documents/git/unet/data/membrane/test")
model = unet()
model.load_weights("unet_membrane.hdf5")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("/home/ay5/Documents/git/unet/data/membrane/test",results)
=======
testGene = testGenerator("data/membrane/test")
model = unet()
model.load_weights("unet_membrane.hdf5")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)
>>>>>>> 6b681003fa361416c54272bbb71ffc113a884692
