## FFmpeg
视频处理最常用的开源软件
功能强大用途广泛，是许多音频与视频的标准解码/编码实现

## 基本概念
### 容器
视频文件本身是一个容器，包括了音频与视频，也可能有字幕等其他内容。

常见的容器格式：
后缀名：
- MP4
- mkv
- WebM
- AVI

```python
ffmpeg -formats
# 可以查看FFmpeg支持的容器 
```
### 编码格式
常用的视频编码格式：
```
H.262
H.264
H.265
```
这三种有版权，但可以免费使用

```
VP8
VP9
AV1
```

这三种无版权

常用音频编码格式
```
MP3
AAC
```
这些格式都是有损压缩，会损失一些细节
以下命令可以查看FFmpeg支持的编码格式，视频编码与音频编码都在内
```
ffmpeg -codecs
```
### 编码器
编码器是实现某种编码格式的库文件，只有安装了某种格式的编码器，才能实现该格式视频/音频的解码和编码。
以下是FFmpeg内置的视频编码器:
```
libx264:最流行的开源H.264编码器
NVENC:基于NVIDIA GPU 的H.264编码器
libx265:开源的HEVC编码器
libvpx:谷歌的VP8和vp9编码器
libbaom:AV1编码器
```
音频编码器：
```
libfdk-aac
aac
```
```
# 查看FFmpeg已经安装的编码器
ffmpeg -encoders
```

## FFmpeg使用格式
命令行参数非常的多
```
ffmpeg {1}{2} -i {3}{4}{5}
```
```
1.全局参数
2.输入文件参数
3.输入文件
4.输出文件参数
5.输出文件
```
参数太多的时候，ffmpeg命令可以写成多行：

```
ffmpeg\
[全局参数]\
[输入文件参数]\
-i [输入文件]\
[输出文件参数]\
[输出文件]
```

eg:
```
ffmpeg\
-y\# 全局参数
-c:a libfdk_acc -c:V libx264\# 输入文件的音频编码格式，视频编码格式
-i input.mps\# 输入文件
-c:v libvpx-vp9 -c:a libvorbis\# 输出文件参数
output.webm 
# 输出文件 
```
若不指明编码格式，FFmpeg回自己判断输入文件的编码。
命令可以简化为:
```
ffmpeg -i input.avi output.mp4
```

## 常用命令行参数
```
-c 指定编码器
-c copy 直接复制
-c:v 指定视频编码器
-c:a 指定音频编码器
-i 指定输入文件
-an 除去音频流
-vn 去除视频流
-preset 指定输出的视频质量，会影响文件的生成速度，有可用值：ultrafast、superfast、veryfast、faster、fast、medium、slow、slower、veryslow
-y 不经过确认，输出时可以直接覆盖同名文件
```
## 常见用法
## 查看文件信息
```python
# 查看视频文件的元信息
ffmpeg -i input.mp4
```

```
# 上面的命令会输出很多冗余的信息，加上-hide_banner参数，可以只显示元信息
ffmpeg -i input.mp4 -hide_banner
```
- 转换编码格式
将视频从一种编码转换为另一种编码，只需指定输出文件的视频编码器即可。
```
ffmpeg -i [input.file] -c:v libx264 output.mp4
```
```
ffmpeg -i [input.file] -c:v libx265 output.mp4
```
## 转换容器格式
将视频文件从一种容器转到另一个容器
```
ffmpeg -i inpyt.mp4 -c copy output.webm
# -c copy直接将输入文件参数复制
```
## 调整码率
改变编码的比特率，用来将视频文件的体积变小。
```
ffmpeg\
-i input.mp4\
-minrate 964k -maxrate 3856k -bufsize 2000k\
output.mp4
# 指定最小、最大码率，缓冲区大小
```
## 改变分辨率
从1080p转到480p
```
ffmpeg\
-i input.mp4\
-vf scale=480:-1\
output.mp4
```
## 提取音频
从视频中提取音频
```
ffmpeg\
-i inpit.mp4\
-vn -c:a copy\
output.aac
```
-vn表示去掉音频，-c:a copy表示不改变音频编码，直接拷贝
## 添加音轨
将外部音频加入视频（有音频与视频两个输入文件）
```
ffmpeg\
-i input.aac -i input.mp4\
output.mp4
```
## 截图
从指定时间开始，连续对1秒钟的视频进行截图
```
ffmpeg\
-y\
-i input.mp4\
-ss 00:01:24 -t 00:00:01\
output_%3d.jpg
```
如果只需要截一张图，可以指定只截取一帧
```
ffmpeg\
-ss 01:23:45\
-i input\
-vframes 1 -q:v 2\
output.jpg
```
-vframes表示只截取一帧，-q:v 2表示输出的图片质量，一般是1-5之间（1为质量最高）

## 剪裁
截取原始视频里面的一个片段，输出为一个新视频，可以指定开始时间和持续时间（duration），也可以指定结束时间(end)。
```
ffmpeg -ss [start] -i [input] -t [duration] -c copy [output]
ffmpeg -ss [start] -i [input] -to [end] -c copy [output]
```
eg；
```
ffmpeg -ss 00:01:50 -i [input] -t 10.5 -c copy [output]
ffmpeg -ss 2.5 -i [intput] -to 10 -c copy [output]
```
## 为音频添加封面
可以为音频添加封面，转为视频，然后上传
```
ffmpeg\
-loop 1\# 表示图片无限循环
-i cover.jpg -i input.mp3\
-c:v libx264 -c:a aac -b:a 192k -shortest\# -shortest表示音频文件结束，输出视频就结束
output.mp4
```







