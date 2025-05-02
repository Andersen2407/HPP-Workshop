import cv2
import numpy as np

img = cv2.imread('Jpeg_exercise\picture.jpg', cv2.COLOR_BGR2RGB)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.waitKey(0)
print("Img shape: ", img.shape)


# 1. Conversion
# Grayscale conversion
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Center around 0
img = np.astype(img, np.int16)
img = img - 128         # Range [0, 255] ==> [-128, 127]
print("Img shape (GRAY): ", img.shape)

# 2. Block partitioning
kh = 8
kw = 8
for h in range(0, img.shape[0], kh):
    for w in range(0, img.shape[1], kw):
        # Copy so its not by reference
        block = img[h:h+kh, w:w+kw].copy()
        
        # Dont do it on the last block if it is not 8x8
        # Bad fix but works for now
        if block.shape != (8, 8):
            continue
        
        # 3. Performing the 2D DCT
        def standard_dct_2d(block: np.ndarray):
            N = block.shape[0]  # Assuming block is 8x8
            
            # Apply 1D DCT on rows
            y_rows = np.zeros_like(block, dtype=np.float32)
            for row in range(block.shape[0]):
                # Extract the row
                x = block[row, :]
                
                # For every y_k
                for k in range(y_rows.shape[1]):
                    # Type-II DCT formula right here! (for one y_k entry)
                    for n in range(y_rows.shape[1]):        # summation happens here
                        y_rows[row, k] += x[n] * np.cos(np.pi*k*(2 * n + 1) / (2*N))
                    y_rows[row, k] *= 2
                # -------------------------------
            
            # Apply 1D DCT on columns
            y_columns = np.zeros_like(block, dtype=np.float32)
            for column in range(block.shape[1]):
                # Apply DCT formula for each column
                x = y_rows[:, column]
                
                for k in range(N):
                    # Type-II DCT formula right here!
                    for n in range(N):
                        y_columns[k, column] += x[n] * np.cos(np.pi*k*(2 * n + 1) / (2*N))
                    y_columns[k, column] *= 2
                # -------------------------------
            
            matrix_y = y_columns
            return matrix_y
        
        matrix_Y = standard_dct_2d(block)
        
        
        # 4. Quantization
        # Quantization matrix for luminance (https://www.sciencedirect.com/topics/computer-science/quantization-table)
        matrix_Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                            [12, 12, 14, 19, 26, 58, 60, 55],
                            [14, 13, 16, 24, 40, 57, 69, 56],
                            [14, 17, 22, 29, 51, 87, 80, 62],
                            [18, 22, 37, 56, 68,109,103,77],
                            [24,35,55,64,81,104,113,92],
                            [49,64,78,87,103 ,113 ,120 ,101],
                            [72 ,92 ,95 ,98 ,112 ,100 ,103 ,99]])
        matrix_Y_Q = np.divide(matrix_Y, matrix_Q)
        matrix_Y_Q = np.round(matrix_Y_Q)
        
        # 5. Reconstruction
        # Inverse quantization
        recovered_block = cv2.dct(matrix_Y_Q, flags=cv2.DCT_INVERSE)
        recovered_block = np.round(recovered_block / 2)
        
        
        # -------- Debugging --------
        # from scipy.fftpack import dct
        # dct_rows = dct(block, axis=1)
        # dct_2d = dct(dct_rows, axis=0)
        # print("DCT 2D\n", dct_2d)
        # print("Matrix Y\n", matrix_Y)
        # exit(0)
        
        
        # Insert the recovered block back into the image
        img[h:h+kh, w:w+kw] = recovered_block
        # print(block[0][:2], matrix_Y_Q[0][:2], recovered_block[0][:2])



print("Final image shape", img.shape)
print(img.dtype)
print(np.max(img), np.min(img))
img = img + 128
print(np.max(img), np.min(img))
img = np.clip(img, 0, 255)  # Ensure values are in the range [0, 255]
img = np.astype(img, np.uint8)  # Convert back to uint8

cv2.imwrite('Jpeg_exercise\compressed.jpg', img)

cv2.imshow('image', img)
cv2.waitKey(0)


