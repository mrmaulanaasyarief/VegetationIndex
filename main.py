import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import tkinter.messagebox
import os

# check image channel, if more than 3 (RGB) than select only the RGB channel
def check_channel(img):
    h, w, ch = img.shape

    if ch > 3:
        image = img[:, :, :3].astype(float)
        image[img[:, :, 3] == 0] = np.nan
        empty_space = img[:, :, 3] == 0
    else:
        image = img
        
    return image
 # Calculate index 
def vegetation_index(img, method):
    # split the each RGB channel
    r = img[:, :, 2].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 0].astype(np.float32)

    if method == "gli":
        # GLI
        vi = np.divide((2 * g - r - b), (2 * g + r + b + 0.00001))
    elif method == "vari":
        # VARI
        vi = np.divide((g - r), (g + r - b + 0.00001))
       
    vi = np.clip(vi, -1, 1)
    return vi

def main():

    while True:
        filename = askopenfilename(title="Select Image File") # show an "Open" dialog box and return the path to the selected file
        if(filename==""):
            if tkinter.messagebox.askretrycancel("Error",  "No Image file selected"):
                pass
            else:
                exit()
        else:
            if(filename.lower().endswith(".jpeg") or filename.lower().endswith(".jpg") or filename.lower().endswith(".png") or filename.lower().endswith(".tiff")):
                break
            else:
                if tkinter.messagebox.askretrycancel("Error",  "Selected file must be in JPEG/JPG/PNG/TIFF format"):
                    pass
                else:
                    exit()

    path =  os.path.dirname(os.path.realpath(__file__))
    print(filename)

    # file berdasarkan selected file
    image_filename = filename

    # read image file to opencv image
    img = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
    image = check_channel(img)

    print('Processing image with shape {} x {}'.format(img.shape[0], img.shape[1]))

    raw = img
    vi = vegetation_index(img, 'gli') #gli or vari
    # print(vi)
    
    index_clear = vi[~np.isnan(vi)]

    # -- Calculate index histogram -- #
    # perc -> values of the histogram bins
    # edges -> the edges of the bins
    perc, edges, _ = plt.hist(index_clear, bins=100, range=(-1, 1), color='darkcyan', edgecolor='black')
    # plt.show()
    plt.close()

    # -- Find the real min, max values of the vegetation_index -- #
    # -- Find the real values of the min, max based on the frequency of the vegetation_index histogramm' s bin in each examined interval -- #
    mask = perc > (0.05 * len(index_clear))
    edges = edges[:-1] 
    min_v = edges[mask].min()
    max_v = edges[mask].max()
    lower, upper = min_v, max_v
    print(lower, upper)

    index_clipped = np.clip(vi, lower, upper)

    # RdYlGn, gray
    cNorm = mpl.colors.Normalize(vmax=upper, vmin=lower)
    cm = plt.get_cmap('RdYlGn')
    colored_image = cm(cNorm(index_clipped))
    
    cm_gray = plt.get_cmap('gray')  
    colored_image_gray = cm_gray(cNorm(index_clipped))

    print(np.sum(colored_image > 0.6))

    figu = plt.figure()
    gs = figu.add_gridspec(2,2)
    axi1 = figu.add_subplot(gs[0])
    axi1.set_title('Raw Image')
    plt.imshow(raw[...,::-1])
    axi3 = figu.add_subplot(gs[1], sharex=axi1, sharey=axi1)
    axi3.set_title('Vegetation Index')
    vi_rgb = cv2.cvtColor(vi, cv2.COLOR_BGR2RGBA)
    plt.imshow(vi_rgb)
    axi2 = figu.add_subplot(gs[2], sharex=axi1, sharey=axi1)
    axi2.set_title('Colored Map')
    ima2 = axi2.imshow(colored_image, cmap=cm)
    caxi = figu.add_axes([axi2.get_position().x1+0.01,axi2.get_position().y0,0.02,axi2.get_position().height])
    plt.colorbar(ima2, cax=caxi) # Similar to fig.colorbar(im, cax = cax)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()
    plt.close()

    img = Image.fromarray(np.uint8(colored_image * 255), mode='RGBA')
    img_gray = Image.fromarray(np.uint8(colored_image_gray * 255), mode='RGBA')

    pil_image_gray = img_gray.convert('RGB') 
    open_cv_image_gray = np.array(pil_image_gray) 
    # Convert RGB to BGR 
    open_cv_image_gray = open_cv_image_gray[:, :, ::-1].copy() 

    pil_image = img.convert('RGB') 
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 


    rgba = np.array(img, dtype=np.float32)

    img.save('result/color_mapped.tiff')
    # img.save('result/result.png')
    np.save('result/result.npy', index_clipped)

    
    mask = cv2.cvtColor(open_cv_image_gray, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(raw, raw, mask=mask)

    _,alpha = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(masked)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    cv2.imwrite("result/masked.png", dst)

    # Visualisasi hasil
    # cv2.imshow("Raw", raw)
    cv2.imwrite("result/raw.png", raw)
    # cv2.waitKey(0)
    # cv2.imshow("VI Result", vi)
    cv2.imwrite("result/vi_result.tiff", vi)
    # cv2.waitKey(0)
    # cv2.imshow("Colored Result", open_cv_image)
    # cv2.waitKey(0)
    # cv2.imshow("Mask", mask)
    cv2.imwrite("result/mask.png", mask)
    # cv2.waitKey(0)
    # cv2.imshow("Masked", masked)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

   
if __name__ == "__main__":
    main()