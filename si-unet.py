# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import array_to_img, img_to_array
import keras as k
import glob, cv2
import data


class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.traindata = "./data/train/"
        self.testdata = "./data/test/"
        self.evaluedata = "./data/test/val/"
        self.result = "./results/"

    def new_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal')(drop5)
        merge6 = k.layers.concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal')(conv6)
        merge7 = k.layers.concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal')(conv7)
        merge8 = k.layers.concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='he_normal')(conv8)
        merge9 = k.layers.concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
        return model

    def train(self):
        model = self.new_unet()
        batch_img = 3
        print("got unet")
        checkpoint_fn = os.path.join('./model/checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5')
        model_checkpoint = ModelCheckpoint(checkpoint_fn, monitor='loss', verbose=1, save_best_only=True)
        tb_cb = TensorBoard(log_dir="./log", write_images=True)
        print('Fitting model...')
        img_num = len(glob.glob(self.traindata + "label\\*.tif"))
        val_num = len(glob.glob(self.evaluedata + "label\\*.tif"))
        model.fit_generator(data.generatedata(self.traindata, batch_img), steps_per_epoch=img_num//batch_img,
                            epochs=20, callbacks=[model_checkpoint, tb_cb],
                            validation_data=data.generatedata(self.evaluedata, batch_img*2),
                            validation_steps=val_num//(batch_img*2)
                            )

    def predict(self):
        model = load_model("./model/checkpoint.09-0.04.hdf5")
        imgsname = glob.glob(self.testdata + "image\\*.tif")
        imgdatas = np.ndarray((1, 512, 512, 3), dtype=np.float32)
        num = 0
        for imgname in imgsname:
            name = imgname[imgname.rindex("\\") + 1:]
            img = cv2.imread(self.testdata + "image\\" + name)
            img = img_to_array(img).astype('float32')
            imgdatas[0] = img / 255
            mask = model.predict_on_batch(imgdatas)
            mask_img = array_to_img(mask[0])
            mask_img.save(self.result + "%s" % name)
            num = num + 1
            if num % 100 == 0:
                print("%d输出完毕！" % num)


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.predict()


