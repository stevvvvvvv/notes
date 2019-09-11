# c++数据类型

[toc]

## 整型

c\++的基本整型分别是`char、short、int、long`和c++11新增的`long long`

其中每种类型都分为有符号版本和无符号版本

- 计算机内存由一些叫做位(bit)的单元组成，各种类型的最小长度如下：

  - `short`至少16位
  - `int`至少与`short`一样长
  - `long`至少32位，且至少与`int`一样长
  - `long long`至少64位，且至少与`long`一样长

- 无符号类型(关键字`unsigned`)：

```c++
unsigned short change;
unsigned int rovert;
unsigned quarterback;   //  有效，因为unsigned是unsigned int的缩写
unsigned long gond;
```

- 整型字面值

`C++`能够以三种不同的计数方式书写整数:十进制，八进制和十六进制。
如果第一位是1~9，则为十进制；如果第一位是0，第二位是1~7，则为八进制；如果第前两位为0x或0X，则为16进制。

- 后缀

后缀是放在数字常量后面的字母，用于表示类型。

整数后面的`l`或`L`表示该数为`long`常量；`u`或`U`表示`unsigned int`常量；`UL`表示`unsigned long`

- `char`类型：字符和小整数

`char`类型是专为存储字符(如字母和数字)设计的。`C++`对字符用单引号，对字符串用双引号。

- `wcha_t`类型和`char16_t`

程序需要处理的字符集无法用一个8位的字节表示时，使用`wchar_t`(宽字符类型)表示扩展字符集。在这种时候需要用`wcin wcout`进行输入输出。当`wchar_t`不够时使用`char16_t`以及`char32_t`，定义方式：

```c++
char16_t ch1 = u'q';
char32_t ch2 = U'\U0000222B';
```

- `bool`类型
`bool is_ready = true;`

## `const`限定符

`const`用来限定常量。

## 浮点数

- 浮点数的书写

```c++
12.34
939001.32
0.00023
8.0
```

以及(E表示法,表示为10的E次方，正数则小数点向右移动，负数则小数点向左移动)

```c++
2.25e+8
8.33E-4
7E5
```

`C++`有三种浮点类型:`float`,`double`和`long double`。
