import numpy as np
from patchify import patchify, unpatchify
#import tifffile as tiff
from PIL import Image
import large_image as lm
import openslide as op
from matplotlib import pyplot as plt
from openslide.deepzoom import DeepZoomGenerator
import staintools


#Let us define a function to detect blank tiles and tiles with very minimal information
#This function can be used to identify these tiles so we can make a decision on what to do with them. 
#Here, the function calculates mean and std dev of pixel values in a tile. 
'''def find_mean_std_pixel_value(img_list):
    
    avg_pixel_value = []
    stddev_pixel_value= []
    for file in img_list:
        #image = tiff.imread(file)
        image = Image.open(file)
        avg = image.mean()
        std = image.std()
        avg_pixel_value.append(avg)
        stddev_pixel_value.append(std)
        
    avg_pixel_value = np.array(avg_pixel_value)  
    stddev_pixel_value=np.array(stddev_pixel_value)
        
    print("Average pixel value for all images is:", avg_pixel_value.mean())
    print("Average std dev of pixel value for all images is:", stddev_pixel_value.mean())
    
    return(avg_pixel_value, stddev_pixel_value)'''


# reference image
reference_tile_loc = "/home/maruvka/Documents/predict_expression/complete_dataset/blk-NGNMRYFIKYRG-TCGA-DC-6681-01Z-00-DX1.png"
reference_tile = staintools.read_image(reference_tile_loc)
# input image
image_name = "TCGA-DJ-A13V-01Z-00-DX1.88661EBD-6B7A-4EF3-95B1-9CAF38B4BCF2"
wsi_path = "/home/maruvka/Downloads/SVS_images/4fb60901-faaa-4fca-b3ca-6e5be4f0ecf6/"+image_name+".svs"
slide = op.open_slide(wsi_path)
#ts = lm.getTileSource(wsi_path)
#ts.getMetadata()

slide_dims = slide.dimensions

#Generating tiles and processing

#Generate object for tiles using the DeepZoomGenerator
tiles = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)
#Here, we have divided our svs into tiles of size 256 with no overlap. 

#The tiles object also contains data at many levels. 
#To check the number of levels
print("The number of levels in the tiles object are: ", tiles.level_count)
print("The dimensions of data in each level are: ", tiles.level_dimensions)
#Total number of tiles in the tiles object
print("Total number of tiles = : ", tiles.tile_count)

###### processing and saving each tile to local directory
cols, rows = tiles.level_tiles[16]


orig_tile_dir_name = "/home/maruvka/Documents/predict_expression/images/saved_tiles/original_tiles"
norm_tile_dir_name = "/home/maruvka/Documents/predict_expression/images/saved_tiles/normalized_tiles/"
H_tile_dir_name = "/home/maruvka/Documents/predict_expression/images/saved_tiles/H_tiles/"
E_tile_dir_name = "/home/maruvka/Documents/predict_expression/images/saved_tiles/E_tiles/"

for row in range(rows):
    for col in range(cols):
        tile_name = image_name+"_"+str(col)+"_"+str(row)
        #tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
        #print("Now processing tile with title: ", tile_name)
        temp_tile = tiles.get_tile(16, (col, row))
        temp_tile_RGB = temp_tile.convert('RGB')
        temp_tile_np = np.array(temp_tile_RGB)
        #Save original tile
        output_image = Image.fromarray(temp_tile_np)
        output_image.save(orig_tile_dir_name+tile_name + "_original.png")
        # standardize brightness
        reference_tile = staintools.LuminosityStandardizer.standardize(reference_tile)
        temp_tile_np = staintools.LuminosityStandardizer.standardize(temp_tile_np)
        
        if temp_tile_np.mean() < 230 and temp_tile_np.std() > 15:
            print("Processing tile number:", tile_name)
            # choose normalization method
            normalizer = staintools.StainNormalizer(method= "macenko")
            # fit to reference tile
            normalizer.fit(reference_tile)
            # normalize the current tile
            norm_img = normalizer.transform(temp_tile_np)
            #norm_img, H_img, E_img = norm_HnE(temp_tile_np, Io=240, alpha=1, beta=0.15)

        #Save the norm tile, H and E tiles      
            output_image = Image.fromarray(temp_tile_np)
            output_image.save(norm_tile_dir_name+tile_name + "_norm.png")
            #tiff.imsave(norm_tile_dir_name+tile_name + "_norm.tif", norm_img)
            #tiff.imsave(norm_tile_dir_name+tile_name + "_norm.tif", norm_img)
            #tiff.imsave(H_tile_dir_name+tile_name + "_H.tif", H_img)
            #tiff.imsave(E_tile_dir_name+tile_name + "_E.tif", E_img)
            
        else:
            print("NOT PROCESSING TILE:", tile_name)