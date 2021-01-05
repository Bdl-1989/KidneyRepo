from common import *
import tifffile as tiff
import json
data_dir = os.getcwd() + '/data'


def read_tiff(image_file):
    image = tiff.imread(image_file)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
        image = np.ascontiguousarray(image)
    return image

# --- rle ---------------------------------
def rle_decode(rle, height, width , fill=255):
    s = rle.split()
    start, length = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    start -= 1
    mask = np.zeros(height*width, dtype=np.uint8)
    for i, l in zip(start, length):
        mask[i:i+l] = fill
    mask = mask.reshape(width,height).T
    mask = np.ascontiguousarray(mask)
    return mask


def rle_encode(mask):
    m = mask.T.flatten()
    m = np.concatenate([[0], m, [0]])
    run = np.where(m[1:] != m[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle =  ' '.join(str(r) for r in run)
    return rle


# --- tile ---------------------------------
def to_tile(image, mask, scale, size, step, min_score):

    half = size//2
    image_small = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    #make score
    height, width, _ = image_small.shape
    vv = cv2.resize(image_small, dsize=None, fx=1 / 32, fy=1 / 32, interpolation=cv2.INTER_LINEAR)
    vv = cv2.cvtColor(vv, cv2.COLOR_RGB2HSV)
    # image_show('v[0]', v[:,:,0])
    # image_show('v[1]', v[:,:,1])
    # image_show('v[2]', v[:,:,2])
    # cv2.waitKey(0)
    vv = (vv[:, :, 1] > 32).astype(np.float32)
    vv = cv2.resize(vv, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

    #make coord
    xx = np.array_split(np.arange(half, width  - half), np.floor((width  - size) / step))
    yy = np.array_split(np.arange(half, height - half), np.floor((height - size) / step))
    # xx = [int(x.mean()) for x in xx]
    # yy = [int(y.mean()) for y in yy]
    xx = [int(x[0]) for x in xx] + [width-half]
    yy = [int(y[0]) for y in yy] + [height-half]


    coord  = []
    reject = []
    for cy in yy:
        for cx in xx:
            cv = vv[cy - half:cy + half, cx - half:cx + half].mean()
            if cv>min_score:
                coord.append([cx,cy,cv])
            else:
                reject.append([cx,cy,cv])
    #-----
    if 1:
        tile_image = []
        for cx,cy,cv in coord:
            t = image_small[cy - half:cy + half, cx - half:cx + half]
            assert (t.shape == (size, size, 3))
            tile_image.append(t)

    if mask is not None:
        mask_small = cv2.resize(mask, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        tile_mask = []
        for cx,cy,cv in coord:
            t = mask_small[cy - half:cy + half, cx - half:cx + half]
            assert (t.shape == (size, size))
            tile_mask.append(t)
    else:
        mask_small = None
        tile_mask  = None

    return {
        'image_small': image_small,
        'mask_small' : mask_small,
        'tile_image' : tile_image,
        'tile_mask'  : tile_mask,
        'coord'  : coord,
        'reject' : reject,
    }


def to_mask(tile, coord, height, width, scale, size, step, min_score):

    half = size//2
    mask  = np.zeros((height, width), np.float32)

    if 0:
        count = np.zeros((height, width), np.float32)
        for t, (cx, cy, cv) in enumerate(coord):
            mask [cy - half:cy + half, cx - half:cx + half] += tile[t]
            count[cy - half:cy + half, cx - half:cx + half] += 1
               # simple averge, <todo> guassian weighing?
               # see unet paper for "Overlap-tile strategy for seamless segmentation of arbitrary large images"
        m = (count != 0)
        mask[m] /= count[m]

    if 1:
        for t, (cx, cy, cv) in enumerate(coord):
            mask[cy - half:cy + half, cx - half:cx + half] = np.maximum(
                mask[cy - half:cy + half, cx - half:cx + half], tile[t] )

    return mask



# --draw ------------------------------------------
def mask_to_inner_contour(mask):
    mask = mask>0.5
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def draw_contour_overlay(image, mask, color=(0,0,255), thickness=1):
    contour =  mask_to_inner_contour(mask)
    if thickness==1:
        image[contour] = color
    else:
        r = max(1,thickness//2)
        for y,x in np.stack(np.where(contour)).T:
            cv2.circle(image, (x,y), r, color, lineType=cv2.LINE_4 )
    return image