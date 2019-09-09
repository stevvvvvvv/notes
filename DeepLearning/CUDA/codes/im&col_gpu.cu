#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__ void im2col(
    const float *data_im,
    const int size,
    const int channels,
    const int height,
    const int width,
    const int filter_h,
    const int filter_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int height_col,
    const int width_col,
    float *data_col,
    int im_stride,
    int col_stride
)
{
    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < size; tid += gridDim.x * blockDim.x)
    {
    //  多batch_size情况
    const int batch_idx = blockIdx.y;
    data_im += batch_idx * im_stride;
    data_col += batch_idx * col_stride;
    
    const int h_index = tid / width_col;                //  纵向tid
    const int h_col = h_index % height_col;             //  纵向第几个
    const int w_col = tid % width_col;                //  横向第几个

    const int c_im = h_index / height_col;              //  第几个通道
    const int c_col = c_im * filter_h * filter_w;       //  col偏移量
    
    const int h_offset = h_col * stride_h - pad_h;      //  纵向开始位置
    const int w_offset = w_col * stride_w - pad_w;      //  横向开始位置
    
    //  channel offset
    float *data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const float *data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;

    for(int i = 0; i < filter_h; ++i)
    {
        for(int j = 0; j < filter_w; ++j)
        {
            int h_im = h_offset + i * dilation_h;
            int w_im = w_offset + j * dilation_w;
            *data_col_ptr = (h_im >= 0 && w_im >=0 && h_im < height && w_im < width)
            ? data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
            // if(h_im >= 0 && w_im >=0 && h_im < height && w_im < width)
            // {
            //     printf("%d \n", data_im_ptr[i * width + j]);
            // }
            data_col_ptr += height_col * width_col;
        }
    }

    // wrong
    // int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // if (tid >= size) return;

    // //  多batch_size情况
    // const int batch_idx = blockIdx.y;
    // data_im += batch_idx * im_stride;
    // data_col += batch_idx * col_stride;

    // //  int tid_copy = tid;
    // //  num表示是第几组，count表示是该组第几个
    // const int channel_num = tid / channel_size;
    // const int channel_count = tid % channel_size;
    // //  定位是第几行的第几个
    // const int w_idx = channel_count % width_col;
    // const int h_idx = channel_count / width_col;
    
    // //  w,h方向开始量
    // const int h_offset = h_idx * stride_h - pad_h;
    // const int w_offset = w_idx * stride_w - pad_w;

    // //  指针引用
    // float *data_im_ptr = data_im;
    // data_im_ptr += (channel_num * height + h_offset) * width + w_offset;
    // float *data_col_ptr = data_col;
    // data_col_ptr += (channel_num * width_col * height_col * height_col + h_idx) * width_col + w_idx;

    // //  copy to col
    // for (int i = 0; i < filter_h; ++i)
    // {
    //     for (int j = 0; j < filter_w; ++j)
    //     {
    //         int h_im = h_offset + i;
    //         int w_im = w_offset + j;
    //         *data_col_ptr = (h_im >= 0 && w_im >=0 && h_im < height && w_im < width)
    //         ? data_im_ptr[i * width + j]
    //         : 0;
    //         data_col_ptr += height_col * width_col;
    //     }
    // }
    }
}

extern "C" __global__ void col2im(
    const float *data_col,
    const int size,
    const int channels,
    const int height,
    const int width,
    const int filter_h,
    const int filter_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int height_col,
    const int width_col,
    float *data_im,
    int im_stride,
    int col_stride
)
{
    for(int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < size; tid += gridDim.x * blockDim.x)
    {
        //  多batch情况
        const int batch_idx = blockIdx.y;
        data_col += batch_idx * col_stride;
        data_im += batch_idx * im_stride;

        float val = 0;
        //  原im中的w,h,c 这里考虑了原im的padding
        const int w_im = tid % width + pad_w;
        const int h_im = (tid / width) % height + pad_h;
        const int c_im = tid / (width * height);
        //  卷积范围
        int kernel_extent_w = (filter_w - 1) * dilation_w + 1;
        int kernel_extent_h = (filter_h - 1) * dilation_h + 1;
        //  compute the start and end of the output
        //  计算横纵方向index，即为col中含有该元素的范围
        const int w_col_start =
            (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);

        const int h_col_start =
            (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        //  到这里就确定了含有tid格元素的col数据范围

        // TODO: use LCM of stride and dilation to avoid unnecessary loops
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
          for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {

            //  kernel位置
            int h_k = (h_im - h_col * stride_h);
            int w_k = (w_im - w_col * stride_w);
            printf("h_im, w_im: %d, %d , tid%d \n", h_im, w_im, tid);
            printf("h_k, w_k: %d, %d , tid%d \n", h_k, w_k, tid);
            printf("h_col, w_col: %d, %d , tid%d \n", h_col, w_col, tid);
            if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
              h_k /= dilation_h;
              w_k /= dilation_w;
              //    w_k * heigth_col的原因:直观理解比较困难，想象一个filter中横向、纵向id相差1个的元素在填到col数据中的内存差距较好理解。
              int data_col_index = (((c_im * filter_h + h_k) * filter_w + w_k) * height_col + h_col) * width_col + w_col;
            // printf("%d \n", data_col_index);
              val += data_col[data_col_index];
            }
          }
        }
        data_im[tid] = val;
      }
    }


int main(){
    using namespace std;
    float *h_data_im;
    float *d_data_im;
    int channels = 3;
    int height = 5;
    int width = 5;
    int filter_h = 3;
    int filter_w = 3;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    float *h_data_col;
    float *d_data_col;

    //  set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    //  计算卷积纵向和横向移动次数
    const int height_col = (height + 2 * pad_h - (dilation_h * (filter_h - 1) + 1)) / stride_h + 1;
    const int width_col = (width + 2 * pad_w - (dilation_w * (filter_w - 1) + 1)) / stride_w + 1;
    const int im_stride = channels * height * width;
    const int col_stride = channels * filter_h * filter_w * height_col * width_col;
    
    //  卷积移动height_col * width_col次，从而data_col有这么多列
    //  每列卷积有channels * filter_h * filter_w个元素

    //  分配CPU空间
    size_t imBytes = channels * height * width * sizeof(float);
    size_t colBytes = channels * width_col * height_col * filter_h * filter_w * sizeof(float);
    h_data_im = (float *)malloc(imBytes);
    h_data_col = (float *)malloc(colBytes);
    memset(h_data_im, 0, imBytes);
    memset(h_data_col, 0, colBytes);

    //  init input data
	for (int m = 0; m < channels; ++m)
	{
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				h_data_im[m * width * height + i * width + j] = m * width * height + i * width + j;
				cout << h_data_im[m * width * height + i * width + j] << ' ';
			}
			cout << endl;
		}
	}

    //  im2col
    //  分配GPU空间
    cudaMalloc((float**)&d_data_im, imBytes);
    cudaMalloc((float**)&d_data_col, colBytes);
    cudaMemcpy(d_data_im, h_data_im, imBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_col, h_data_col, colBytes, cudaMemcpyHostToDevice);

    //  总线程数
    const int im2col_batch_size = 1;
    const int im2col_BLOCK_SIZE = 256;
    const int im2col_size = height_col * width_col * channels;

    dim3 im2col_dim_grid(ceil((float)im2col_size / im2col_BLOCK_SIZE), im2col_batch_size);
    im2col<<<im2col_dim_grid, im2col_BLOCK_SIZE>>>(d_data_im, im2col_size,
        channels, height, width, 
        filter_h, filter_w, 
        pad_h, pad_w,
        stride_h, stride_w, 
        dilation_h, dilation_w, 
        height_col, width_col, d_data_col,
        im_stride, col_stride);

    //  copy back
    cudaMemcpy(h_data_col, d_data_col, colBytes, cudaMemcpyDeviceToHost);
    
    //  print
    for(int i = 0; i < filter_w * filter_h * channels; ++i)
    {    
        for(int j = 0; j < height_col * width_col; ++j)
        {
            cout << h_data_col[i * height_col * width_col + j] << ' ';
        }
        cout <<endl;
    }

    //  cuda free
    cudaFree(d_data_im);
    cudaFree(d_data_col);

    //  col2im
    //  host free
    free(h_data_im);

    //  分配GPU空间
    cudaMalloc((float**)&d_data_im, imBytes);
    cudaMalloc((float**)&d_data_col, colBytes);
    cudaMemcpy(d_data_im, h_data_im, imBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_col, h_data_col, colBytes, cudaMemcpyHostToDevice);

    //  总线程数
    const int col2im_batch_size = 1;
    const int col2im_BLOCK_SIZE = 256;
    const int col2im_size = height * width * channels;

    dim3 col2im_dim_grid(ceil((float)col2im_size / col2im_BLOCK_SIZE), col2im_batch_size);
    col2im<<<col2im_dim_grid, col2im_BLOCK_SIZE>>>(d_data_col, col2im_size,
        channels, height, width, 
        filter_h, filter_w, 
        pad_h, pad_w,
        stride_h, stride_w, 
        dilation_h, dilation_w, 
        height_col, width_col, d_data_im,
        im_stride, col_stride);
    
    //  copy back
    cudaMemcpy(h_data_im, d_data_im, imBytes, cudaMemcpyDeviceToHost);

    //  print
    for (int m = 0; m < channels; ++m)
	{
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				cout << h_data_im[m * width * height + i * width + j] << ' ';
			}
			cout << endl;
		}
	}

    //  device free
    cudaMalloc((float**)&d_data_im, imBytes);
    cudaMalloc((float**)&d_data_col, colBytes);
    cudaMemcpy(d_data_im, h_data_im, imBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_col, h_data_col, colBytes, cudaMemcpyHostToDevice);

    //  host free
    free(h_data_im);
    free(h_data_col);

    cin.get();
    cin.get();
    return 0;
}
