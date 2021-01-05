import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()

import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from common  import *
from dataset import *
from model   import *

import torch.cuda.amp as amp
is_mixed_precision = False  # True #True #



def run_submit():

    fold = 2
    out_dir = os.getcwd() + '/submit/result//fold%d'%fold
    #out_dir = '/root/share1/kaggle/2020/hubmap/result/resnet34/fold-all'
    initial_checkpoint = \
        out_dir+'/checkpoint/00010000_model.pth' #

    #server = 'local' # local or kaggle
    server = 'kaggle'


    #---
    net = Net().cuda()
    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)['state_dict']
    net.load_state_dict(state_dict,strict=True)  #True
    net = net.eval()

    #---
    if server == 'local':
        valid_image_id = make_image_id('valid-%d' % fold)
    if server == 'kaggle':
        valid_image_id = make_image_id('test-all')


    tile_size = 600 #320
    tile_average_step = 192 #192
    tile_scale = 0.25
    tile_min_score = 0.25


    start_timer = timer()
    for id in valid_image_id:
        if server == 'local':
            image_file = data_dir + '/train/%s.tiff' % id
            image = read_tiff(image_file)
            mask_file = data_dir + '/train/%s.mask.png' % id
            #mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) #pixels <= CV_IO_MAX_IMAGE_PIXELS
            mask = np.array(PIL.Image.open(mask_file))#.convert('RGB')

        if server == 'kaggle':
            image_file = data_dir + '/test/%s.tiff' % id
            image = read_tiff(image_file)
            mask = None


        #--- predict here!  ---
        tile = to_tile(image, mask, tile_scale, tile_size, tile_average_step, tile_min_score)

        tile_image = tile['tile_image']
        tile_image = np.stack(tile_image)[..., ::-1]
        tile_image = np.ascontiguousarray(tile_image.transpose(0,3,1,2))
        tile_image = tile_image.astype(np.float32)/255

        tile_probability = []
        batch = np.array_split(tile_image, len(tile_image)//4)
        for t,m in enumerate(batch):
            print('\r %s  %d / %d   %s'%(id, t, len(batch), time_to_str(timer() - start_timer, 'sec')), end='',flush=True)
            m = torch.from_numpy(m).cuda()

            net(m)
            p = []
            with torch.no_grad():
                logit = data_parallel(net, m)
                p.append(torch.sigmoid(logit))

                #---
                if 1: #tta here
                    logit = data_parallel(net, m.flip(dims=(2,)))
                    p.append(torch.sigmoid(logit.flip(dims=(2,))))

                    logit = data_parallel(net, m.flip(dims=(3,)))
                    p.append(torch.sigmoid(logit.flip(dims=(3,))))
                #---

            p = torch.stack(p).mean(0)
            tile_probability.append(p.data.cpu().numpy())

        print('')
        #print('%s : %s' %(id, time_to_str(timer() - start_timer, 'sec')))
        tile_probability = np.concatenate(tile_probability).squeeze(1)

        height, width = tile['image_small'].shape[:2]
        probability = to_mask(tile_probability, tile['coord'], height, width,
                      tile_scale, tile_size, tile_average_step, tile_min_score)

        #--- show results ---
        if server == 'local':
            truth = tile['mask_small'].astype(np.float32)/255
        if server == 'kaggle':
            truth = np.zeros((height, width), np.float32)

        overlay = np.dstack([
            np.zeros_like(truth),
            probability, #green
            truth, #red
        ])
        image_small = tile['image_small'].astype(np.float32)/255
        overlay1 = 1-(1-image_small)*(1-overlay)

        predict = (probability>0.5).astype(np.float32)

        image_show_norm('image_small', image_small, min=0, max=1, resize=0.1)
        image_show_norm('probability', probability, min=0, max=1, resize=0.1)
        image_show_norm('predict',     predict, min=0, max=1, resize=0.1)
        image_show_norm('overlay',     overlay,     min=0, max=1, resize=0.1)
        image_show_norm('overlay1',    overlay1,    min=0, max=1, resize=0.1)
        cv2.waitKey(1)


        if server == 'kaggle':
            cv2.imwrite(out_dir+'/valid/%s.predict.png'%id, (predict*255).astype(np.uint8))
            cv2.imwrite(out_dir+'/valid/%s.image_small.png'%id, (image_small*255).astype(np.uint8))
            cv2.imwrite(out_dir+'/valid/%s.overlay1.png'%id, (overlay1*255).astype(np.uint8))
            cv2.imwrite(out_dir+'/valid/%s.overlay.png'%id, (overlay*255).astype(np.uint8))
            cv2.imwrite(out_dir+'/valid/%s.probability.png'%id, (probability*255).astype(np.uint8))

        #---

        if server == 'local':
            pass

            loss = np_binary_cross_entropy_loss(probability, truth)
            dice = np_dice_score(probability, truth)
            tp, tn = np_accuracy(probability, truth)
            print('loss',loss)
            print('dice',dice)
            print('tp, tn',tp, tn)
            print('')
            cv2.waitKey(0)


        zz=0

def run_make_csv():

    fold = 2
    out_dir = os.getcwd() + '/submit/result//fold%d'%fold
    #out_dir = '/root/share1/kaggle/2020/hubmap/result/resnet34/fold-all'
    csv_file = \
        out_dir+'/submit-fold-2-resnet34-00010000_model.csv' #

    #-----
    image_id = make_image_id('test-all')
    predicted = []

    for id in image_id:
        image_file = data_dir + '/test/%s.tiff' % id
        image = read_tiff(image_file)

        height, width = image.shape[:2]
        predict_file = out_dir+'/valid/%s.predict.png'%id
        #predict = cv2.imread(predict_file, cv2.IMREAD_GRAYSCALE)
        predict = np.array(PIL.Image.open(predict_file))
        predict = cv2.resize(predict, dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        predict = (predict>128).astype(np.uint8)*255

        p = rle_encode(predict)
        predicted.append(p)

    df = pd.DataFrame()
    df['id'] = image_id
    df['predicted'] = predicted

    df.to_csv(csv_file, index=False)
    print(df)


# main #################################################################
if __name__ == '__main__':
    #run_submit()
    run_make_csv()

'''
 

'''