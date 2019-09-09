#include <iostream>
// #include <sys/time.h>

bool is_a_ge_zero_and_a_lt_b(int a,int b)
{
    if(a>=0 && a <b) return true;
    return false;
}

// inline double seconds()
// {
//     struct timeval tp;
//     struct timezone tzp;
//     int i = gettimeofday(&tp, &tzp);
// //    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
//     return ((double)tp.tv_usec);
// }

// void im2col_cpu(const float* data_im,
//     const int channels,
//     const int height,
//     const int width,
//     const int kernel_h,
//     const int kernel_w,
//     const int pad_h,
//     const int pad_w,
//     const int stride_h,
//     const int stride_w,
//     const int dilation_h,
//     const int dilation_w,
//     float* data_col) {
//   const int output_h = (height + 2 * pad_h -
//     (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
//   const int output_w = (width + 2 * pad_w -
//     (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
//   const int channel_size = height * width;
//   for (int channel = channels; channel--; data_im += channel_size) {
//     for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
//       for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
//         int input_row = -pad_h + kernel_row * dilation_h;
//         for (int output_rows = output_h; output_rows; output_rows--) {
//           if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
//             for (int output_cols = output_w; output_cols; output_cols--) {
//               *(data_col++) = 0;
//             }
//           } else {
//             int input_col = -pad_w + kernel_col * dilation_w;
//             for (int output_col = output_w; output_col; output_col--) {
//               if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
//                 *(data_col++) = data_im[input_row * width + input_col];
//               } else {
//                 *(data_col++) = 0;
//               }
//               input_col += stride_w;
//             }
//           }
//           input_row += stride_h;
//         }
//       }
//     }
//   }
// }

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;//计算卷积层输出图像的高
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;//计算卷积层输出图像的宽
  const int channel_size = height * width;//计算卷积层输入单通道图像的数据容量

  /*第一个for循环表示输出的矩阵通道数和卷积层输入图像通道是一样的，每次处理一个输入通道的信息*/
  /*正反写法在这里没有关系，后面的计算不涉及channel*/
  for (int channel = channels; channel--; data_im += channel_size) {

	/*第二个和第三个for循环表示了输出单通道矩阵的某一列，同时体现了输出单通道矩阵的行数*/
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {

        int input_row = -pad_h + kernel_row * dilation_h;//在这里找到卷积核中的某一行在输入图像中的第一个操作区域的行索引
        
		/*第四个和第五个for循环表示了输出单通道矩阵的某一行，同时体现了输出单通道矩阵的列数*/
        /*同上，这里的正反循环也和后面计算没有关系*/
        for (int output_rows = output_h; output_rows; output_rows--) {
            
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {//如果计算得到的输入图像的行值索引小于零或者大于输入图像的高(该行为pad)
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;//那么将该行在输出的矩阵上的位置置为0
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;//在这里找到卷积核中的某一列在输入图像中的第一个操作区域的列索引
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {//如果计算得到的输入图像的列值索引大于等于于零或者小于输入图像的宽(该列不是pad)
                *(data_col++) = data_im[input_row * width + input_col];//将输入特征图上对应的区域放到输出矩阵上
              } else {//否则，计算得到的输入图像的列值索引小于零或者大于输入图像的宽(该列为pad)
                *(data_col++) = 0;//将该行该列在输出矩阵上的位置置为0
              }
              input_col += stride_w;//按照宽方向步长遍历卷积核上固定列在输入图像上滑动操作的区域
            }
          }
          input_row += stride_h;//按照高方向步长遍历卷积核上固定行在输入图像上滑动操作的区域
        }
      }
    }
  }
}

// void col2im_cpu(const float* data_col,
// const int channels,
// const int height,
// const int width,
// const int height_col,
// const int width_col,
// const int kernel_h,
// const int kernel_w,
// const int pad_h,
// const int pad_w,
// const int stride_h,
// const int stride_w,
// const int dilation_h,
// const int dilation_w,
// float* data_im
// )
// {
//   const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
//   const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
//   const int channel_size = height * width;

//   for (int channel = channels; channel--; data_im += channel_size)
//   {
//     for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
//     {
//       for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
//       {
//         int input_row = kernel_row * dilation_h - pad_h;
//         for (int output_rows = output_h; output_rows; output_rows--)
//         {
//           if (!is_a_ge_zero_and_a_lt_b(input_row, height))
//           {
//             data_col += output_w;
//           }
//           else
//           {
//             int input_col = kernel_col * dilation_w - pad_w;
//             for (int output_col = output_w; output_col; output_col--)
//             {
//               if (is_a_ge_zero_and_a_lt_b(input_col, width))
//               {
//                 data_im[input_row * width + input_col] += *data_col;
//               }
//               data_col++;
//               input_col += stride_w;
//             }
//           }
//           input_row += stride_h;
//         }
//       }
//     }
//   }
// }

/*col2im_cpu为im2col_cpu的逆操作接收13个参数，分别为输入矩阵数据指针(data_col)，卷积操作处理的一个卷积组的通道
数(channels)，输入图像的高(height)与宽(width)，原始卷积核的高(kernel_h)与宽(kernel_w)，
输入图像高(pad_h)与宽(pad_w)方向的pad，卷积操作高(stride_h)与宽(stride_w)方向的步长，
卷积核高(stride_h)与宽(stride_h)方向的扩展，输出图像数据指针(data_im)*/
template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;//计算卷积层输出图像的宽
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;//计算卷积层输出图像的高
  const int channel_size = height * width;//col2im输出的单通道图像容量
  for (int channel = channels; channel--; data_im += channel_size) {//按照输出通道数一个一个处理
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;//在这里找到卷积核中的某一行在输入图像中的第一个操作区域的行索引
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {//如果计算得到的输入图像的行值索引小于零或者大于输入图像的高(该行为pad)
            data_col += output_w;//那么，直接跳过这output_w个数，这些数是输入图像第一行上面或者最后一行下面pad的0
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;//在这里找到卷积核中的某一列在输入图像中的第一个操作区域的列索引
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {//如果计算得到的输入图像的列值索引大于等于于零或者小于输入图像的宽(该列不是pad)
                data_im[input_row * width + input_col] += *data_col;//将矩阵上对应的元放到将要输出的图像上
              }//这里没有else，因为如果紧挨的if条件不成立的话，input_row * width + input_col这个下标在data_im中不存在，同时遍历到data_col的对应元为0
              data_col++;//遍历下一个data_col中的数
              input_col += stride_w;//按照宽方向步长遍历卷积核上固定列在输入图像上滑动操作的区域
            }
          }
          input_row += stride_h;//按照高方向步长遍历卷积核上固定行在输入图像上滑动操作的区域
        }
      }
    }
  }
}

template <typename Dtype>
void col2im_coef_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_coef) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;

  // for (int channel = channels; channel--; data += channel_size) {

  //   for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
  //     for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {

  //       int input_row = -pad_h + kernel_row * dilation_h;
        
  //       for (int output_rows = output_h; output_rows; output_rows--) {
  //         if (is_a_ge_zero_and_a_lt_b(input_row, height)){
  //           int input_col = -pad_w + kernel_col * dilation_w;
  //           for (int output_col = output_w; output_col; output_col--) {
  //             if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
  //               *(data_coef++) += 1;
  //             } 
  //             input_col += stride_w;
  //           }
  //         }
  //         input_row += stride_h;
  //       }
  //     }
  //   }
  // }

    for (int channel = channels; channel--; data_coef += channel_size) {//按照输出通道数一个一个处理
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;//在这里找到卷积核中的某一行在输入图像中的第一个操作区域的行索引
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {//如果计算得到的输入图像的行值索引小于零或者大于输入图像的高(该行为pad)
            data_col += output_w;//那么，直接跳过这output_w个数，这些数是输入图像第一行上面或者最后一行下面pad的0
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;//在这里找到卷积核中的某一列在输入图像中的第一个操作区域的列索引
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {//如果计算得到的输入图像的列值索引大于等于于零或者小于输入图像的宽(该列不是pad)
                data_coef[input_row * width + input_col] += 1;//将矩阵上对应的元放到将要输出的图像上
              }//这里没有else，因为如果紧挨的if条件不成立的话，input_row * width + input_col这个下标在data_im中不存在，同时遍历到data_col的对应元为0
              data_col++;//遍历下一个data_col中的数
              input_col += stride_w;//按照宽方向步长遍历卷积核上固定列在输入图像上滑动操作的区域
            }
          }
          input_row += stride_h;//按照高方向步长遍历卷积核上固定行在输入图像上滑动操作的区域
        }
      }
    }
  }
}

int main()
{
    using namespace std;
    float *data_im;
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
    float *data_col;
    int channels = 3;

    double iStart, iElaps;

    const int output_h = (height + 2 * pad_h - (dilation_h * (filter_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (filter_w - 1) + 1)) / stride_w + 1;
    data_im = new float[channels * height * width]; //  开辟相应字节的空间数
    data_col = new float[channels * output_h * output_w * filter_h * filter_w];

    //  init input data
    for (int m = 0; m < channels; ++m)
    {
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                data_im[m * width * height + i * width + j] = m * width * height + i * width + j;
                cout << data_im[m * width * height + i * width + j] << " ";
            }
            cout << endl;
        }
    }

    cout << endl;
    cout << endl;

    // iStart = seconds();
    im2col_cpu(data_im,
    channels,
    height,
    width,
    filter_h,
    filter_w,
    pad_h,
    pad_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    data_col);
    // iElaps = seconds() - iStart;
    // cout << "col2im cost: " << iElaps << "secodes" << endl;


    cout << channels << endl;
    cout << output_h << endl;
    cout << output_w << endl;
    cout << filter_h << endl;
    cout << filter_w << endl;
   // cout <<"error"<<endl;
    for(int i = 0; i < filter_w * filter_h * channels; ++i)
    {
        for(int j = 0; j < output_w * output_h; ++j)
        {
            cout << data_col[i * output_w * output_h + j] << ' ';
        }
        cout <<endl;
    }

    cout << endl;
    cout << endl;

    float *data_im_trans;
    data_im_trans = new float[channels * height * width]; //  开辟相应字节的空间数

    // iStart = seconds();
    col2im_cpu(data_col,
    channels,
    height,
    width,
    filter_h,
    filter_w,
    pad_h,
    pad_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    data_im_trans);
    // iElaps = seconds() - iStart;
    // cout << "col2im cost: " << iElaps << "secodes" << endl;
    //  col2im的结果是放大过后的

  for (int m = 0; m < channels; ++m)
    {
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                cout << data_im_trans[m * width * height + i * width + j] << " ";
            }
            cout << endl;
        }
    }
  
  cout << endl;
  cout << endl;

  //  求col2im膨胀的系数
  float *data_coef;
  data_coef = new float[channels * height * width];

  col2im_coef_cpu(data_col,
    channels,
    height,
    width,
    filter_h,
    filter_w,
    pad_h,
    pad_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    data_coef);

  for (int m = 0; m < channels; ++m)
    {
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                cout << data_coef[m * width * height + i * width + j] << " ";
            }
            cout << endl;
        }
    }

  cout << endl;
  cout << endl;

  for (int m = 0; m < channels; ++m)
    {
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {   
              data_im_trans[m * width * height + i * width + j] /= data_coef[m * width * height + i * width + j];
              cout << data_im_trans[m * width * height + i * width + j] << " ";
            }
            cout << endl;
        }
    }

  return 0;
}
