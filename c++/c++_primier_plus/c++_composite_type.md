##### 数组

数组(`array`)能够存储多个同类型的值，创建数组使用声明语句，指出：

-   存储在每个元素中的值的类型
-   数组名
-   数组中的元素数

声明的通用格式如下：

`typeName arrayName[arraySize];`

例：

`short months[12];`

这说明我们创建了一个包含`12`个`short`元素的数组`months`。

声明数组还可以用如下的格式：

`int hands[4] = {5, 6, 7, 9}`

##### 字符串

`c++`处理字符串的方式有两种，第一种为C-风格字符串，另一种基于`string`类库。

-   C-风格字符串：

将字符串存储在`char`数组中，以空字符串结尾，写作`\0`，其ASCII码为0，用来标记字符串的结尾。

`char cat[8] = {'f', 'a', 't', 'e', 's', 's', 'a', '\0'}`

或者(这样调用会自动在最后加上或者补全长度个`'\0'`，记得预留空间)：
```
char bird[11] = "Mr. Cheeps";
char fish[] = "Bubbles";
```

-   在数组中使用字符串
    -   将数组初始化为字符串常量
    -   将键盘或文件输入读入到数组中

```
#include <iostream>
#include <cstring>

int main()
{
    using namespace std;
    const int Size = 15;
    char name1[Size];
    char name2[Size] = "C++owboy";  //  有的编译器会要求static关键字
    
    cout << "Howdy! I'm" << name2 << "!"
    " What's your name?\n";
    cin >> name1;
    cout << "Well, " << name1 << ", your name has "
    cout << strlen(name1) << " letters and is stored\n";
    cout << "in array of " << sizeof(name1) << " bytes.\n";
    cout << "Your initial is " << name1[0] << ".\n"
    name2[3] = '\0';
    cout << "Here are the first 3 characters of my name: ";
    cout << name2 << endl;
    return 0;
}
```
`cin`仅能读取一个单词，因为它碰到空格就会默认停止，空格后面的输入会保留在`cin`队列中，在下次进行调用。

例：
```
cin >> first_input;
//  keyboard input:First Seconde
cin >> seconde_input;
//  first_input = First, seconde_input = Second
```

-   读取一行字符串输入

`iostream`中的`cin`类提供一些面向行的函数:`getline()`和`get()`,这两种方法都读取一行输入直到到达换行符。`getline()`丢弃换行符更改为空字符，而`get()`保留。

例：
```
cin.getline(name, 20);
```

如果使用`cin.get()`，由于不丢弃换行符，在第二次调用时会自动检测到队列中的换行符并终止，这种情况可以使用下面两种调用方法：
```
cin.get(name, Size);
cin.get();
cin.get(dessert, Size);

//  or:
cin.get(name, Size).get()
```
第二种方法是因为`cin.get(name, Size)`返回了一个`cin`对象，所以可以再次调用`get()`函数，同样的，我们也可以：

`cin.getline(name1, Size).getline(name2, Size)`

使用`cin.clear()`可以清空`cin`队列。

-   `string`类简介

`string str = "panther";`

`string`类不需要指定长度，在第一次赋值时会自动调整长度。

`string`类的IO与`char`类有所区别：
```
using namespace std;
char charr[20];
string str;
cin.getline(charr, 20);
getline(cin, str);
```

##### 结构

结构是一种比数组更灵活的数据格式，一个结构可以存储多种类型的数据。

结构是用户定义的类型，形式如下：
```
struct inflatable
{
    char name[20];
    float volume;
    double price;
};
```
`struce`说明这是一个结构布局，`inflatable`是这种数据格式的名称，大括号内包含结构存储的数据类型的列表，每个列表项称为结构成员。

定义了结构后，可以创建这种类型的变量：
```
inflatable hat;
inflatable woopie;
struct inflatable mainframe;
```
创建了变量后，就可以使用`.`来进行访问：
```
hat.volume;
hat.price;
```
其他的一些初始化方法：
```
struct perks
{
    int key_number;
    char car[12];
} mr_smith, ms_jones;

//  or
struce perks
{
    int key_number;
    char car[12];
} mr_glitz = 
{
    7,
    "Packard"
};
```

-   结构数组

创造元素为结构的数组：

`inflatable gifts[100];`

则`gifts`是一个`inflatable`数组，每个元素都是一个`inflatable`对象。

初始化结构数组：
```
inflatable guests[2] = 
{
    {"Bambi", 0.5, 21.99};
    {"Godzilla", 2000, 565.99};
};
```

##### 共用体

共用体(`union`)是一种数据格式，能够存储不同的数据类型，但只能同时存储其中的一种类型。
```
union one4all
{
    int int_val;
    long long_val;
    double double_val;
};
```
可以使用`one4all`存储`int`、`long`或`double`：
```
one4all pail;
pail.int_val = 15;
pail.double_val = 1.38;
```
因此，`pail`有时可以是`int`，有时可以是`double`。共用体的用途之一是当数据项使用两种或更多种格式（但不会同时使用时）节省空间以及内存。

##### 枚举

- `c++`的`enum`工具提供了另一种创建符号常量的方式，可以代替`const`还可以定义新类型：
`enum spectrum {red, orange, yellow, green, blue, violet, indigo, ultraviolet};`
这条语句完成两项工作：

  -   让`spectrum`成为新类型的名称；`spectrum`被称为枚举。
  -   将`red, orange, yellow`等作为符号常量，它们对应整数`0~7`，称为枚举量。

可以使用枚举名来声明这种类型的变量：
```
spectrum band;
```
在不进行强制类型转换的情况下，只能将定义枚举时使用的枚举量赋给这种枚举的变量：
```
band = blue;
```
因此`spectrum`变量受到限制，只有8个可能的值。枚举更常被用来定义相关的符号变量，而不是新类型。

-    设置枚举量的值

可以用以下方式给枚举量赋值：
```
enum bits{one = 1, two = 2, four = 4, eight = 8};
enum bigstep{first, second = 100, third};
enum {zero, null = 0, one, numero_uno = 1}; //  这种情况下zero=null=0, one=numero_uno=1
```

-   枚举量的取值范围

取值范围的定义：首先找出上限，需要知道枚举量的最大值。找到大于这个最大值的、最小的2的幂，将其减去1得到的便是取值范围的上限。例如`bigstep`中最大值为100，则上限值为$128 - 1 = 127$。要计算下限需要找到最小值，如果它不小于0，那么取值范围的下限为0，否则与寻找上限的方法相同，但是加上负号。例如最小值为-6，则比它小的、最大的2次幂是-8，因此下限是$-8 + 1 = 7$。

##### 指针和自由存储空间

-   计算机程序在存储数据时必须跟踪的3种基本类型：

    -   信息存储在何处
    -   存储的值为多少
    -   存储的信息是什么类型

定义变量的另一种策略是以指针为基础，指针是一个变量，它存储的是值的地址而不是值本身。

在讨论指针前我们先看看如何找到常规变量的地址，只需要使用`&`：
```
#include <iostream>
int main()
{
    using namespace std;
    int donuts = 6;
    double cups = 4.5;

    cout << "donuts value = " << donuts;
    cout << " and donuts address = " << &donuts << endl;
    cout << "cups value = " << cups;
    cout << " and cups address = " << &cups << endl;
    return 0; 
}
```

显示地址时使用十六进制表示法，这是常用于描述内存的表示法，两个地址值的差可能是4或8，这是`int`及`double`的长度。

使用常规变量时，值是指定的量，而地址是派生量。使用指针存储数据的方法刚好相反：将地址视为指定的量，而将值视为派生量。一种特殊类型的变量——指针用于存储值的地址，因此指针名表示的是地址，`*`运算符被称为间接值或接触引用运算符，将其用于指针可以得到该地址处存储的值。例：
```
#include <iostream>
int main()
{
    using namespace std;
    int updates = 6;
    int *p_updates;
    p_updates = &updates;

    cout << "Values: updates = " << updates;
    cout << ", *p_updates = " << *p_updates << endl;
    
    *p_updates = *p_updates + 1;
    cout << "Now updates = " << updates << endl;
    return 0;
}
```

-   声明和初始化指针

指针声明必须指定指针指向的数据的类型，如：
```
int *p_updates;
double *tax_ptr;
char *str;
```
这说明`*p_updates`的类型是`int`，而`p_updates`为一个指向`int`的指针,而`tax_ptr, str`分别是指向`double, char`的指针。

可以在声明语句中初始化指针，在这种情况下被初始化的是指针，而不是它指向的值：
```
int higgens = 5;
int * pt = &higgens;
```
这段语句将`pt`(而不是`*pt`)的值设置为`&higgens`

指针的初始化方法：
```
#include <iostream>
int main()
{
    using namespace std;
    int higgens = 5;
    int *pt = &higgens;

    cout << "Value of higgens = " << higgens
    << "; Address of higgens = " << &higgens << endl;
    cout << "Value of *pt = " << *pt
    << "; Value of pt = " << pt << endl;
    return 0;
}
```

-   指针的危险

`c++`中创建指针时，计算机将分配用来存储地址的内存，