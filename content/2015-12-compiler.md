Title: Something You Should Kown About C/C++ Compiler
Date: 2015-12-07
Slug: compiler
Category: Technology


我以前经常问身边的同学关于 C/C++ 代码编译链接的问题，问他们知不知道这方面的细节。非常遗憾，绝大部分人连编译和链接都分不清楚，感慨学校教的 C 语言课太水了，很多同学也不怎么敲代码=。=，我写这篇文章记录一下我自己对 C/C++ 编译器的理解，希望也能帮助到其他同学，希望你们在遇到编译链接错误时，不要慌张，冷静分析，找对排查问题的方向。

我们都知道编译代码的过程分成 4 个环节，预处理，编译，汇编，链接（生成可执行文件或者库文件）。首先我们需要明确几点。

1. 头文件只参与预处理环节，跟编译和后续环节没有什么关系。
2. 任何一个源文件都是独立编译，相互之间不干扰。
3. 前三个环节并不涉及到第三方的库文件，但是在预处理环节会涉及到库的头文件。
4. 预处理环节一般不会出错，哪怕你使用了一个未定义的宏，也不会出错，只会在编译时报错，而任何编译错误都不会跟库文件扯上任何关系。
5. 汇编一般也不会出错，它只是编译器一个中间的隐性环节，把汇编代码编译成目标代码。
6. 链接是天坑，标准库和第三方库的库文件都是在这个环节加入进来的。
7. 我们遇到的问题，基本上都是编译出错或者链接出错。然而事情并没有那么简单。

### 预处理

预处理我们接触最多的就是包含头文件的 `#include` 和定义宏的 `#define`，同时还有规范性的头文件保护宏。在这个环节出现最多的问题是找不到头文件，然后编译器停止工作。这个问题是最容易被解决的，但是很多人可能并不是很了解编译器寻找头文件的流程。这里我们首先要区分 3 种头文件，第一种是标准库的头文件，第二种是代码中引用到的第三方库的头文件，第三种是自己代码编写的头文件。编译器在工作时是有一个头文件目录列表的，根据目录列表去寻找头文件，第一个目录便是当前代码的所在的目录，其次是编译器自行定义的目录（一般是标准库头文件所在的目录），最后是我们自己在编译时加上的目录列表，可以有多个，包括需要引用的第三方库的头文件目录和自己代码的头文件目录（可能你自己写的头文件跟源文件不在一个目录下）。

### 编译与汇编

编译环节出错那基本就是语法的问题，代码写得有问题直接导致编译器停止工作。这里不得不提一下 C++11 标准，并不是所有编译器都实现了新标准的所有规范，同时可能编译器之间实现的特性还有差异，这些在写跨编译器或跨平台代码事要特别注意，同一个编译器的不同版本之间也会略有差异。

### 链接

链接环节出现问题非常之多，类型也是五花八门，奇奇怪怪，会非常坑。但是归根结底主要是两种，第一是链接时找不到符号，第二是链接时找到了多个符号。很多同学碰到链接出错时编译器吐出来的一堆一堆乱七八糟的函数符号估计都很蛋疼，但是我们多数时候碰到的都是 `undefined reference to xxx`，即找不到符号。

##### 符号

要搞明白链接时编译器是怎么工作的，我们就得先搞清楚 `符号` 在编译系统中的作用。一个符号可以指代一块内存或者一段代码。代码中与符号相关的几处地方如下。

1. 变量的声明，告诉编译器有这么一个变量指代一块内存。
2. 变量的定义，告诉编译器需要为这个变量分配一块内存。
3. 函数的声明，告诉编译器有这么一段代码可以使用，输入输出规范如何，应该怎么调用。
4. 函数的定义，告诉编译器这段代码的逻辑实现。
5. 引用变量或函数，代码中使用某个变量或者调用某个函数。

```c++
extern int a; // declare
int a; // define

void foo(); // declare
void foo() { // define
  a = 0;
}
```

编译器会给每个变量和每个函数分配一个符号，这样做的好处是方便符号的重用（函数的重用），也利于项目代码的模块化，多个目标文件的链接。由于每个源文件代码都是独立编译的，并生成目标文件，编译器在处理这个源文件时，最后会在目标文件中指出它所需要的符号和它能够提供的符号，这样，链接器在链接一堆目标文件时（库所提供的目标文件和自己代码的目标文件）就能够为每个待确定的符号找到对应的符号，从而成功生成可执行文件或者库文件。

##### 找不到符号

这个问题估计大部分同学在自己编译代码的时候都碰到过，绝大多数情况下都是编译时配置出错，没有告诉链接器应该去链接某个文件，而导致找不到符号。然而在有时候已经完全配置好了，还是会出现这种情况，即我知道这个库的符号全在这个目标文件中或者这个静态库或者这个动态库中，但是编译器还是报错说找不到符号。这种情况带出了一些更深层次的问题。

##### C 与 C++ 其实并没有想象中那么和谐

我们都知道 C++ 可以重载函数，类似于下面的这段代码。

```c++
void foo(int x) {
  x = 0;
}

void foo(float x) {
  x = 0.f;
}
```

C 中是不允许函数重名的，但是 C++ 中可以通过不同的输入参数类型和类型次序来重载同名函数，暂且不论重载带来的好处和坑，C++ 能这么做是因为 C++ 编译器会重写每个函数最后生成的符号，上面两个函数在编译完后会生成不同的符号，这样一来，对链接器来说其实函数名相同已经没有什么意义了。

```
vagrant@trusty64:~/test$ cat a.cpp
void foo(int x) {
    x = 0;
}

void foo(float x) {
    x = 0.f;
}
vagrant@trusty64:~/test$ gcc -c a.cpp -o a.o
vagrant@trusty64:~/test$ nm a.o
0000000000000010 T _Z3foof
0000000000000000 T _Z3fooi
vagrant@trusty64:~/test$

```

我们可以看到 gcc 编译出来的符号已经跟函数名不一样了，符号包含了更多的信息，比如符号的类型（这个符号是个函数），和函数对应参数的类型，相当复杂。

```
vagrant@trusty64:~/test$ cat a.c
void foo(int x) {
    x = 0;
}
vagrant@trusty64:~/test$ gcc -c a.c -o a.o
vagrant@trusty64:~/test$ nm a.o
0000000000000000 T foo
vagrant@trusty64:~/test$
```

相对来说 C 代码编译出来的符号是和函数名是一致的，同时符号中也不区分变量和函数。

以上的代码只是很简单的函数，如果加上命名空间，类函数等，编译器产生的符号会更加复杂，更加吓人，这也是为什么我们看到的链接出错中会有一长串的字符，因为 C++ 中的符号异常复杂，包含的信息太多。

编译器对 C 和 C++ 代码的处理方式不同，导致两者采用完全不一样的符号命名机制，这样会造成很多链接时的问题，所有我们可以看到好多 C 语言库的头文件里会写下面这种代码。

```c++
#ifdef __cplusplus
extern "C" {
#endif

......
......

#ifdef __cplusplus
}
#endif
```

通过这种方式告诉编译器，这个头文件的所有符号请按照 C 语言的规则进行生成，不要采用 C++ 那套符号重写机制。如果不采取这种措施，就会导致原本在库中是 `foo` 的符号被改写成 `_Z3fooi` 类似的形式而造成链接失败。

以上这种问题本质上是编译器产生的符号于实际库中的符号不一致。C 和 C++ 不同的符号机制会导致这种情况，但是还有其他问题也会导致这种情况，事实上，一般 C 库的作者在这方面都考虑到了的，在头文件中设置宏是可以解决这种问题的，而且这个不需要使用库的人自行设定。

##### C++ 编译器中符号的兼容性

多数情况下，我们使用的第三方库都是库的提供者事先编译好的，这就带来了一个很大的隐患。同一个函数在库中的符号和我们编译器要寻找的符号可能不一致，这个问题在 MSVC 上尤为突出。除去动态库静态库的差异，针对相同编译器的不同版本，同一个函数可能生成的符号会不一样，这是最最坑爹的地方。看看 OpenCV 里 VC10，VC11，VC12 的各个目录就知道这个差异是非常大的。相对而言，gcc 不同版本之间的兼容性似乎就好很多。当然 C 的代码相对于 C++ 就好很多了，不可能出现这种坑爹的情况。

### 总结

胡说八道了一堆，希望能够帮助你理解 C/C++ 代码编译时出现的问题，从而针对性地寻求相应的解决方案。
