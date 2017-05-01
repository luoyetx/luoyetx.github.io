Title: Redis 源码之简单动态字符串
Date: 2015-03-14
permalink: reading-redis-1
Category: Technology


Redis 并没有使用 C 中的字符数组，而是自己实现了一个简单动态字符串 SDS 结构。Redis 利用一块连续内存空间实现了动态字符串（字符串也是以'\0'结束，与部分 C 字符串操作函数兼容）。与 sds 相关的源码在 sds.h 和 sds.c 两个文件中。

#### SDS 结构定义

![redis-sdshr]({filename}/images/2015/redis-sdshr.jpg)

Redis 中定义了 sds 类型，实际为 char* 类型，结构体 sdshdr 可以理解为 sds 类型的头信息。sdshdr 和 sds 在内存中是一个整体，由 zmalloc(Redis 自己实现的一个 malloc 版本) 申请得到。

```c
typedef char *sds; // sds 类型，实际等于 sdshdr 结构体中的 buf

struct sdshdr {
    unsigned int len; // 整个字节数组buf的长度，实际 buf 指向的内存区域有 len+1 个字节（'\0'永远占一个字节）
    unsigned int free; // buf 中剩余的字节数
    char buf[]; // 指向字节数组的指针
};
```

#### SDS 基本操作函数

##### sds 字符串的创建与释放

Redis 在初始化 sds 字符串时就会为 buf 申请一块连续的内存，紧跟在 buf 后面。


```c
/* Create a new sds string with the content specified by the 'init' pointer
 * and 'initlen'.
 * If NULL is used for 'init' the string is initialized with zero bytes.
 * 利用 init 指针所指向的内容和 initlen 指定的长度来初始化 sds 字符串
 * 如果 init 为 NULL，则用字节'\0'来填充 sds 字符串
 *
 * The string is always null-termined (all the sds strings are, always) so
 * even if you create an sds string with:
 * 这个字符串永远都是以 NULL 结尾，哪怕你用下面的方式调用
 *
 * mystring = sdsnewlen("abc",3);
 *
 * You can print the string with printf() as there is an implicit \0 at the
 * end of the string. However the string is binary safe and can contain
 * \0 characters in the middle, as the length is stored in the sds header.
 * 由于字符串以'\0'结尾，你可以使用 printf() 来打印字符串，但字符串本身是二进制安全的，
 * 可以存放字节0，实际存放的字节数在 sds 头信息中 */
sds sdsnewlen(const void *init, size_t initlen) {
    struct sdshdr *sh;

    // 申请一整块内存，内存前几个字节存放 sdshdr 信息，后面的即为字节数组
    if (init) {
        sh = zmalloc(sizeof(struct sdshdr)+initlen+1);
    } else {
        sh = zcalloc(sizeof(struct sdshdr)+initlen+1); // 以字节0填充
    }
    if (sh == NULL) return NULL;
    sh->len = initlen;
    sh->free = 0;
    if (initlen && init)
        memcpy(sh->buf, init, initlen); // 复制数据
    sh->buf[initlen] = '\0';
    return (char*)sh->buf;
}
```

释放 sds 字符串的内存十分简单，因为 sdshdr 和 sds 为一个整体，因此释放时之需提供 sdshdr 的地址，即 `sds - sizeof(struct sdshdr)`。释放的函数 zfree 也是 Redis 自己实现的 free 版本

```c
/* Free an sds string. No operation is performed if 's' is NULL. */
void sdsfree(sds s) {
    if (s == NULL) return;
    zfree(s-sizeof(struct sdshdr));
}
```

##### sds 字符串的动态调整

sds 字符串最关键的就是它的长度的动态调整，包括字节数组的拓展和压缩。

Redis 提供了一个函数用来拓展 sds 字符串的长度，这个函数用来拓展一个 sds 字符串的空余长度，由输入参数 addlen 控制，表示需要有 addlen 个空余字节数。实际 Redis 在操作时会判断字符串的长度，如果拓展后的总长度（已用和未用的字节数）少于 1M，则新的总长度为拓展后的 2 倍，否则新的总长度为拓展后总长度加 1M。这样做的目的就是为了减少 Redis 内存分配的次数，下次需要拓展时可能就不需要分配内存了。

```c
/* Enlarge the free space at the end of the sds string so that the caller
 * is sure that after calling this function can overwrite up to addlen
 * bytes after the end of the string, plus one more byte for nul term.
 *
 * Note: this does not change the *length* of the sds string as returned
 * by sdslen(), but only the free buffer space we have. */
sds sdsMakeRoomFor(sds s, size_t addlen) {
    struct sdshdr *sh, *newsh;
    size_t free = sdsavail(s);
    size_t len, newlen;

    if (free >= addlen) return s; // 已有的空闲字节数大于要求的字节数，没必要拓展
    len = sdslen(s);
    sh = (void*) (s-(sizeof(struct sdshdr))); // sdshdr 指针
    newlen = (len+addlen);
    if (newlen < SDS_MAX_PREALLOC) // SDS_MAX_PREALLOC = 1024*1024
        newlen *= 2;
    else
        newlen += SDS_MAX_PREALLOC;
    newsh = zrealloc(sh, sizeof(struct sdshdr)+newlen+1); // '\0'需要固定的一个字节
    if (newsh == NULL) return NULL; // 分配失败

    newsh->free = newlen - len; // 更新空闲长度，不包括'\0'
    return newsh->buf;
}
```

sds 字符串既然有拓展，当然也会有压缩。Redis 使用了 sdsRemoveFreeSpace 函数用来回收 sds 字符串中的所有空余空间，实际操作时 Redis 会重新分配一块内存，将原有的数据复制到新的内存区域，并释放原有空间。

```c
/* Reallocate the sds string so that it has no free space at the end. The
 * contained string remains not altered, but next concatenation operations
 * will require a reallocation.
 * 重新分配 sds 字符串的空间，使它没有空余空间。
 * 字符串中的内容不会改变，但是下次作字符串连接操作时，又会重新分配内存
 *
 * After the call, the passed sds string is no longer valid and all the
 * references must be substituted with the new pointer returned by the call.
 * 函数调用后，输入的指针 s 将会变成无效的 */
sds sdsRemoveFreeSpace(sds s) {
    struct sdshdr *sh;

    sh = (void*) (s-(sizeof(struct sdshdr)));
    sh = zrealloc(sh, sizeof(struct sdshdr)+sh->len+1); // 重新分配内存
    sh->free = 0;
    return sh->buf;
}
```

##### sds 字符串的操作函数

sds 字符串除了部分兼容 C 的字符串操作函数，Redis 自身也实现了一些 sds 字符串操作函数，包括字符串连接函数 sdscat 系列，整数转换成字符串，格式化字符串，字符串分割等很多操作。

sdscat 系列函数中最基础的就是 `sds sdscatlen(sds s, const void *t, size_t len)` 函数，它将指针 t 指向的内存空间的 len 个字节连接到 s 字符串上。其他的 cat 操作都是基于这个函数的。

```c
/* Append the specified binary-safe string pointed by 't' of 'len' bytes to the
 * end of the specified sds string 's'.
 *
 * After the call, the passed sds string is no longer valid and all the
 * references must be substituted with the new pointer returned by the call. */
sds sdscatlen(sds s, const void *t, size_t len) {
    struct sdshdr *sh;
    size_t curlen = sdslen(s);

    s = sdsMakeRoomFor(s,len); // 分配足够的内存空间
    if (s == NULL) return NULL;
    sh = (void*) (s-(sizeof(struct sdshdr))); // 内存变更过后的 sdshdr 指针
    memcpy(s+curlen, t, len); // 复制
    sh->len = curlen+len;
    sh->free = sh->free-len;
    s[curlen+len] = '\0';
    return s;
}
```

`sds sdstrim(sds s, const char *cset)` 实现了字符串左右的 trim 操作。`void sdsrange(sds s, int start, int end)` 实现了字符串的子串操作，支持负索引。`void sdstolower(sds s)` 和 `void sdstoupper(sds s)` 调用了 C 内置的大小写转换函数用来实现 sds 字符串的大小写转换。Redis 还实现了其他很多字符串操作，很值得学习。

#### SDS 小结

sds 字符串是 Redis 中最基础的结构，使用一块连续内存来存放字符串的元信息和数据，减少了内存操作，申请与释放都只需要一次操作，而有了字符串的元信息，很多字符串的操作就能够得到优化，最简单的例子就是计算字符串的长度，C 中的 strlen 函数复杂度为 O(n)，而 Redis 中计算长度只是 O(1) 的操作。SDS 结构只是 Redis 中最基础的数据结构，不依赖其他 Redis 的数据结构，源码相对简单，但其中的设计仍然值得好好体味和学习。
