# c++复合类型

[toc]

## 数组

数组(`array`)能够存储多个同类型的值，创建数组使用声明语句，指出：

- 存储在每个元素中的值的类型
- 数组名
- 数组中的元素数

声明的通用格式如下：

`typeName arrayName[arraySize];`

例：

`short months[12];`

这说明我们创建了一个包含`12`个`short`元素的数组`months`。

声明数组还可以用如下的格式：

`int hands[4] = {5, 6, 7, 9}`

## 字符串

`c++`处理字符串的方式有两种，第一种为C-风格字符串，另一种基于`string`类库。

- C-风格字符串：

将字符串存储在`char`数组中，以空字符串结尾，写作`\0`，其ASCII码为0，用来标记字符串的结尾。

`char cat[8] = {'f', 'a', 't', 'e', 's', 's', 'a', '\0'}`

或者(这样调用会自动在最后加上或者补全长度个`'\0'`，记得预留空间)：

```c++
char bird[11] = "Mr. Cheeps";
char fish[] = "Bubbles";
```

- 在数组中使用字符串
  - 将数组初始化为字符串常量
  - 将键盘或文件输入读入到数组中

```c++
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

```c++
cin >> first_input;
//  keyboard input:First Seconde
cin >> seconde_input;
//  first_input = First, seconde_input = Second
```

- 读取一行字符串输入

`iostream`中的`cin`类提供一些面向行的函数:`getline()`和`get()`,这两种方法都读取一行输入直到到达换行符。`getline()`丢弃换行符更改为空字符，而`get()`保留。

例：

```c++
cin.getline(name, 20);
```

如果使用`cin.get()`，由于不丢弃换行符，在第二次调用时会自动检测到队列中的换行符并终止，这种情况可以使用下面两种调用方法：

```c++
cin.get(name, Size);
cin.get();
cin.get(dessert, Size);

//  or:
cin.get(name, Size).get()
```

第二种方法是因为`cin.get(name, Size)`返回了一个`cin`对象，所以可以再次调用`get()`函数，同样的，我们也可以：

`cin.getline(name1, Size).getline(name2, Size)`

使用`cin.clear()`可以清空`cin`队列。

- `string`类简介

`string str = "panther";`

`string`类不需要指定长度，在第一次赋值时会自动调整长度。

`string`类的IO与`char`类有所区别：

```c++
using namespace std;
char charr[20];
string str;
cin.getline(charr, 20);
getline(cin, str);
```

## 结构

结构是一种比数组更灵活的数据格式，一个结构可以存储多种类型的数据。

结构是用户定义的类型，形式如下：

```c++
struct inflatable
{
    char name[20];
    float volume;
    double price;
};
```

`struce`说明这是一个结构布局，`inflatable`是这种数据格式的名称，大括号内包含结构存储的数据类型的列表，每个列表项称为结构成员。

定义了结构后，可以创建这种类型的变量：

```c++
inflatable hat;
inflatable woopie;
struct inflatable mainframe;
```

创建了变量后，就可以使用`.`来进行访问：

```c++
hat.volume;
hat.price;
```

其他的一些初始化方法：

```c++
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

- 结构数组

创造元素为结构的数组：

`inflatable gifts[100];`

则`gifts`是一个`inflatable`数组，每个元素都是一个`inflatable`对象。

初始化结构数组：

```c++
inflatable guests[2] =
{
    {"Bambi", 0.5, 21.99};
    {"Godzilla", 2000, 565.99};
};
```

## 共用体

共用体(`union`)是一种数据格式，能够存储不同的数据类型，但只能同时存储其中的一种类型。

```c++
union one4all
{
    int int_val;
    long long_val;
    double double_val;
};
```

可以使用`one4all`存储`int`、`long`或`double`：

```c++
one4all pail;
pail.int_val = 15;
pail.double_val = 1.38;
```

因此，`pail`有时可以是`int`，有时可以是`double`。共用体的用途之一是当数据项使用两种或更多种格式（但不会同时使用时）节省空间以及内存。

## 枚举

- `c++`的`enum`工具提供了另一种创建符号常量的方式，可以代替`const`还可以定义新类型：
`enum spectrum {red, orange, yellow, green, blue, violet, indigo, ultraviolet};`
这条语句完成两项工作：

  - 让`spectrum`成为新类型的名称；`spectrum`被称为枚举。
  - 将`red, orange, yellow`等作为符号常量，它们对应整数`0~7`，称为枚举量。

可以使用枚举名来声明这种类型的变量：

```c++
spectrum band;
```

在不进行强制类型转换的情况下，只能将定义枚举时使用的枚举量赋给这种枚举的变量：

```c++
band = blue;
```

因此`spectrum`变量受到限制，只有8个可能的值。枚举更常被用来定义相关的符号变量，而不是新类型。

- 设置枚举量的值

可以用以下方式给枚举量赋值：

```c++
enum bits{one = 1, two = 2, four = 4, eight = 8};
enum bigstep{first, second = 100, third};
enum {zero, null = 0, one, numero_uno = 1}; //  这种情况下zero=null=0, one=numero_uno=1
```

- 枚举量的取值范围

取值范围的定义：首先找出上限，需要知道枚举量的最大值。找到大于这个最大值的、最小的2的幂，将其减去1得到的便是取值范围的上限。例如`bigstep`中最大值为100，则上限值为$128 - 1 = 127$。要计算下限需要找到最小值，如果它不小于0，那么取值范围的下限为0，否则与寻找上限的方法相同，但是加上负号。例如最小值为-6，则比它小的、最大的2次幂是-8，因此下限是$-8 + 1 = 7$。

## 指针和自由存储空间

- 计算机程序在存储数据时必须跟踪的3种基本类型：

  - 信息存储在何处
  - 存储的值为多少
  - 存储的信息是什么类型

定义变量的另一种策略是以指针为基础，指针是一个变量，它存储的是值的地址而不是值本身。

在讨论指针前我们先看看如何找到常规变量的地址，只需要使用`&`：

```c++
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

```c++
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

- 声明和初始化指针

指针声明必须指定指针指向的数据的类型，如：

```c++
int *p_updates;
double *tax_ptr;
char *str;
```

这说明`*p_updates`的类型是`int`，而`p_updates`为一个指向`int`的指针,而`tax_ptr, str`分别是指向`double, char`的指针。

可以在声明语句中初始化指针，在这种情况下被初始化的是指针，而不是它指向的值：

```c++
int higgens = 5;
int * pt = &higgens;
```

这段语句将`pt`(而不是`*pt`)的值设置为`&higgens`

指针的初始化方法：

```c++
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

- 指针的危险

`c++`中创建指针时，计算机将分配用来**存储地址**的内存，但不会分配用来存储**所指向的数据**的内存。如下：

```c++
long *fellow;
*fellow = 223323;
```

`fellow`的确是一个指针，但是代码没有将地址赋给`fellow`，由于`fellow`没有被初始化，它可以是任何值。

- 指针和数字

指针不是整型，不能简单地将整数赋给指针，要将数字值作为地址来用，应当通过强制类型转换将数字转换为适当的地址类型：

```c++
ing * pt;
pt = (int *)0xB8000000;
```

这样赋值语句的两边都是整数的地址，是一个有效的赋值。

- 使用`new`来分配内存

通过使用指针可以在运行阶段分配未命名的内存以存储值，在这种情况下只能通过指针访问内存。我们可以通过`malloc()`或者`new`来分配内存。

```c++
int * pn = new int;
typeName * pointer_name = new typeName;
```

`new int`告诉程序需要适合存储`int`的内存，`new`运算符根据类型确定需要内存的字节数，然后找到这样的内存并返回地址，接下来将地址赋值给`pn`。

```c++
#include <iostream>
int main()
{
    using namespace std;
    int nights = 1001;
    int * pt = new int;
    *pt = 1001;

    cout << "nights value = "
    << nights << ": location " << &nights << endl;
    cout << "int value = " << *pt << ": location = " << pt << endl;

    double * pd = new double;
    *pd = 10000001.0;

    cout << "double " << "value = " << *pd << ": location = " << pd << endl;
    cout << "location of pd:" << &pd << endl;
    return 0;
}
```

- 使用`delete`释放内存

```c++
int * ps = new int;
//  ... some action
delete ps;
```

`delete`将释放`ps`指向的内存，但不会删除`ps`指针本身，一定要配对地使用`new`和`delete`，否则会发生内存泄露。

- 使用`new`创建动态数组

如果程序只需要一个值，可能会通过声明一个简单变量的方法；对于大型数据（数组、字符串和结构）是`new`的用武之地。

假设程序需要一个数组，但是这个数组的元素是有条件读取的。如果我们在一开始就声明数组，给它分配相应的空间，这种方法叫做静态联编；但使用`new`时，如果在运行阶段需要数组，则创建它，这样可以在程序运行时选择数组长度的方法叫做动态联编，意味着数组是在程序运行时创建的，这种数组叫做动态数组。

在`c++`中使用`new`创建动态数组：
`int * psome = new int [10];`
`new`运算符返回第一个元素的地址。当程序使用完`new`分配的内存块时，应使用`delete`释放他们：
`delete [] psome;`
方括号告诉程序应释放整个数组，而不仅仅是指针指向的元素。方括号的使用如下：

```c++
int * pt = new int;
delete [] pt;
short * ps = new short [500];
delete ps;
```

使用`new`和`delete`时应当遵循：

  1. 不要使用`delete`释放不是`new`分配的内存
  2. 不要使用`delete`释放同一个内存块两次
  3. 如果使用`new []`为数组分配内存，则应使用`delete []`来释放
  4. 如果使用`new []`为一个实体分配内存，则应使用`delete`来释放
  5. 对空指针应用`delete`是安全的

为数组分配内存的通用格式：
`type_name * pointer_name = new type_name [num_elements];`

使用动态数组：

`int * psome = new int [10];`
可以看作是一根指向数组第一个元素的手指，可以使用`psome[n]`访问第`n`个元素。

```c++
#include <iostream>
int main()
{
    using namespace std;
    double * p3 = new double [3];
    p3[0] = 0.2;
    p3[1] = 0.5;
    p3[2] = 0.8;
    cout << "p3[1] is " << p3[1] << ".\n";
    p3 = p3 + 1;
    cout << "Now p3[0] is " << p3[0] << " and ";
    cout << "p3[1] is " << p3[1] << ".\n";
    p3 = p3 - 1;
    delete [] p3;
    return 0;
}
```

`p3 = p3 + 1`不能修改数组名的值，但是可以修改指针的值，最后`delete`前将指针值的修改重置以便于给`delete []`提供正确的地址。

## 指针、数组和指针算术

指针和数组基本等价的原因在于指针算术和`c++`内部处理数组的方式。将整数变量加1后，其值将增加1；将指针变量加1后，增加的量等于它指向的类型的字节数。

在多数情况下，`c++`将数组名解释为数组第1个元素的地址。下面的语句将`pw`声明为指向`double`类型的指针，然后将它初始化为`wages`：
`double * pw = wages;`
和所有数组一样，`wages`也存在以下关系：
`pw = &wages[0] = address of first element of array`
且有如下形式的转换：
`arrayname[i] becomes *(arrayname + i);    //  对于数组`
`pointername[i] becomes *(pointername + i); //  对于指针`
区别在于指针地址可以进行整数加法，而数组不行：

```c++
pointername = pointername + 1;  //  有效
arrayname = arrayname + 1;  //  无效
```

且对数组的`sizeof`得到的是数组的长度，而对指针的`sizeof`得到的是指针的长度。

初始化数组指针应当用以下方式：
`short (*pas) [20] = &tell;`

- 指针小结

1. 声明指针

    要声明指向特定类型的指针，应当使用以下方式：

    ```c++
    typeName * pointerName;
    double * pn;
    char * pc;
    ```

2. 给指针赋值

    应将内存地址赋值给指针，例：

    ```c++
    double * pn;
    double * pa;
    char * pc;
    double bubble = 3.2;
    pn = &bubble;
    pc = new char;
    pa = new double[30];
    ```

3. 对指针解除引用

    ```c++
    cout << *pn;
    *pc = 'S';
    ```

    对指针应用解除引用或间接值运算符(`*`)来解除引用。

4. 区分指针和指针所指向的值

    ```c++
    int * pt = new int;
    *pt = 5；
    ```

    这里`pt`是指向`int`的指针，而`*pt`的类型为`int`而非指针。

5. 数组名

    在多数情况下，`c++`将数组名视为数组的第一个元素的地址。

6. 指针算术

    ```c++
    int tacos [10] = {5, 2, 8, 4, 1, 2, 2, 4, 6, 8};
    int * pt = tacos;
    pt = pt + 1;
    int *pe = &tacos[9];
    pe = pe - 1;
    int diff = pe - pt; //  return 7
    ```

7. 数组的动态联编和静态联编

    ```c++
    int tacos[10];  //  静态联编

    int size;
    cin >> size;
    int * pz = new int [size];  //  动态联编
    ...
    delete [] pz;   //  释放内存
    ```

8. 数组表示法和指针表示法

    使用方括号数组表示法等同于对指针解除引用：

    ```c++
    tacos[0] means *tacos means the value at address tacos
    tacos[3] means *(tacos + 3) means the value at address tacos + 3
    ```

    指针名和指针变量都是如此，因此对于指针和数组名，既可以用指针表示法，也可以使用数组表示法：

    ```c++
    int * pt = new int [10];
    *pt = 5;
    pt[0] = 6;
    pt[9] = 44;
    int coats[10];
    *(coats + 4) = 12;
    ```

- 指针和字符串

```c++
char flower[10] = "rose";
cout << flower << "s are red\n";

const char* bird = "fly";
cout << bird << endl;

cout << flower << " at " << (int *) flower << endl;
cout << bird << " at " << (int *) bird << endl;
```

字符串的用法和指针类似，字符串字面值是常量，所以声明中加入关键字`const`。

一般来说，如果给`cout`提供一个地址，它将打印一个地址。如果指针的类型为`char *`，则`cout`将显示指向的字符串，如果要显示字符串的地址，则需要`(int *)`这样的强制转换。

如下的方式只是复制地址而不是复制字符串：

```c++
char * ps;
char[20] animal = "fox";
ps = animal;
```

要复制字符串需要如下方式：

```c++
#include <cstring>
ps = new char[strlen(animal) + 1];
strcpy(ps, animal);
```

这样就可以复制字符串，`ps`指向一个新的地址。

**应使用`strcpy()`或`strncpy()`而不是赋值运算符来将字符串赋给数组**

- 使用`new`创建动态结构

在运行时创建数组优于在编译时创建数组，对于结构也是如此。将`new`用用于结构分为两步：

1. 创建结构

    要创建结构，需要同时使用结构类型和`new`：
    `inflatable * ps = new inflatable;`

2. 访问结构成员

    创建动态结构时，不能将成员运算符句点用于结构名。`c++`专门为这种情况提供了一个运算符：箭头成员运算符(`->`)，用于指向结构的指针，就像点运算符作用于结构名一样。例如：

    ```c++
    struct things
    {
        int good;
        int bad;
    };
    things grubnose = {3, 453};
    things * pt= &grubnose;
    pt -> good == 3;
    pt -> bad == 453;
    ```

    另一种访问结构成员的方法是，如果`ps`是指向结构的指针，则`*ps`就是被指向的结构本身，由于`*ps`是一个结构，则可以使用`(*ps).price`这样的方法来访问。

    一个使用`new`和`delete`节省空间的示例：

    ```c++
    #include <iostream>
    #include <cstring>
    using namespace std;
    char* getname(void);
    int main()
    {
        char* name;
        name = getname();
        cout << name << " at " << (int*)name << endl;
        delete[] name;
        name = getname();
        cout << name << " at " << (int*)name << endl;
        delete[] name;
        return 0;
    }

    char* getname()
    {
        char temp[80];
        cout << "Enter last name: ";
        cin >> temp;
        char* pn = new char[strlen(temp) + 1];
        strcpy(pn, temp);
        return pn;
    }
    ```

    `getname()`将输入的单词放到`temp`数组中，然后使用`new`分配内存储存单词。

- 自动存储、静态存储和动态存储

根据用于分配内存的方法，`c++`有三种管理数据内存的方式：自动存储、静态存储和动态存储（有时也叫自由存储空间或堆）。

1. 自动存储

    在函数内部定义的常规变量使用自动存储空间，被称为自动变量，它们在所属的函数被调用时产生，在该函数结束时消亡。上面`getname()`函数中的`temp`便是这样的变量，当程序控制权回到`main()`时，`temp`使用的内存将自动被释放。

    自动变量通常存储在**栈**中，这意味着执行代码块时，其中的变量将依次加入到栈中，而在离开代码块时，将按相反的顺序释放这些变量，这被称为**后进先出(LIFO)**.

2. 静态存储

    静态存储是整个程序执行期间都存在的存储方式。使变量成为静态的方式有两种，一种是在函数外面定义它，一种是在声明变量时使用关键字`static`。

3. 动态存储

    `new`和`delete`运算符提供了一种比自动变量和静态变量更灵活的方法。它们管理了一个内存池，在`c++`中称为自由存储空间或**堆**。`new`和`delete`可以在一个函数中分配内存，而在另一个函数中释放它。

## 类型组合

可以以各种方式组合数组、结构和指针。从结构开始：

```c++
struct antarctica_years_end
{
    int year;
};
```

然后创造这种类型的变量：

```c++
antarctica_years_end s01, s02, s03;
```

然后使用成员运算符访问成员：

```c++
s01.year = 1998;
```

可创建指向这种结构的指针：

```c++
antarctica_years_end * pa = &s02;
```

将该指针设置为有效地之后，就可以使用间接成员运算符访问成员：

```c++
pa -> year = 1999;
```

可以创建结构数组：

```c++
antarctica_years_end trio [3];  //  arrays of 3 structures
```

然后可以使用成员运算符访问元素的成员：

```c++
trio[0].year = 2003;
```

其中`trio`是一个数组，`trio[0]`是一个结构，而`trio[0].year`是该结构的一个成员。由于数组名是一个指针，因此也可以使用间接成员运算符：

```c++
(trio+1)->year = 2004;
```

可以创建指针数组：

```c++
const antarctica_years_end * arp[3] = {&s01, &s02, &s03};
```

`arp`是一个指针数组，`arp[1]`是一个指针可以使用间接成员运算符访问：

```c++
std::cout << arp[1]->year << std::endl;
```

可以创建指向上述数组的指针：

```c++
const antarctica_years_end ** ppa = arp;
```

其中`arp`是一个数组的名称，因此它是第一个元素的地址。但它的第一个元素为指针，因此`ppa`是一个指向指针的指针，在`c++11`中在混淆时可以通过`auto`推断`ppb`的类型：

```c++
auto ppb = arp;
```

使用`ppa`访问数组：

```c++
std::cout << (*ppa)->year << std::endl;
std::cout << (*(ppb+1))->year << std::endl;
```

## 数组的替代品

- 模板类`vector`

模板类`vector`类似于`string`类，也是一种动态数组。`vector`类使用`new`和`delete`管理内存，`vector`类示例如下：

```c++
#include <vector>                   //  要使用vector类必须包含头文件vector
...
using namespace std;                //  vector包含在名称空间std中
vector<int> vi;                     //  create a zero_size array of int
int n;
cin >> n;
vector<double> vd(n);               //  create an array of n doubles
```

其中`vi`是一个`vector<int>`对象，`vd`是一个`vector<double>`对象，由于`vector`对象在插入或添加值的时候自动调整长度，所以`vi`的初始长度可以为零。创建`vector`类的通法：

```c++
vector<typeName> vt(n_elem);
```

这里的`n_elem`可以是整数变量，也可以是整数常量。

- 模板类`array`(`c++11`)

`vector`类的功能比数组强大，但是效率稍低。如果需要的是长度固定的数组，使用数组是更佳的选择，代价是不那么方便和安全。因此`c++11`新增了模板类`array`，他也位于`std`中。`array`对象的长度和数组一样是固定的，也使用**栈**而不是自由存储区，因此效率与数组相同但更安全、方便。

创建`array`对象的方法：

```c++
#include <array>
...
using namespace std;
array<int, 5> ai;                   //  create array object of 5 ints
array<double, 4> ad = {1.2, 2.1, 3.43, 4.3};
```

创建`array`的通法：

```c++
array<typeName, n_elem> arr;
```

这里的`n_elem`只可以是整数常量。

- 比较数组、`vector`对象和`array`对象

```c++
#include <iostream>
#include <vector>
#include <array>
int main()
{
    using namespace std;
    //  C, original C++
    double a1 [4] = {1.2, 2.4, 3.6, 4.8};

    //  C++98 STL
    vector<double> a2(4);
    //  no simple way to initialize in C98
    a2[0] = 1.0/3.0;
    a2[1] = 1.0/5.0;
    a2[2] = 1.0/7.0;
    a2[3] = 1.0/9.0;

    //  C++11 -- create and initialize array object
    array<double, 4> a3 = {3.14, 2.72, 1.62, 1.41};
    array<double, 4> a4;
    a4 = a3;

    //  use array notation
    cout << "a1[2]: " << a1[2] << " at " << &a1[2] << endl;
    cout << "a2[2]: " << a2[2] << " at " << &a2[2] << endl;
    cout << "a3[2]: " << a3[2] << " at " << &a3[2] << endl;
    cout << "a4[2]: " << a4[2] << " at " << &a4[2] << endl;

    //  misdeed
    a1[-2] = 20.2;                      //  这种情况会造成数组越界，可能影响其他的值
    cout << "a1[-2]: " << a1[-2] << " at " << &a1[-2] << endl;
    cout << "a3[2]: " << a3[2] << " at " << &a3[1] << endl;
    cout << "a4[2]: " << a4[2] << " at " << &a4[1] << endl;
    return 0;
}
```

为了防止数组越界，我们可以使用成员函数`at()`：

```c++
a2.at(1) = 2.3;
```
