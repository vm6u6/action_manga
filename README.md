# Action Manga

* Detection text:
- Roboflow api (YOLOv8 for now)
- DBNet ++ (TODO)

* OCR:
- Manga-OCR (japanese only)

* Translate:
- M2M-100
- ChatGPT (TODO)

[After Translate PART 1]
* relocate the text on the original img

[After Translate PART 2]
* Animate:
- ?


## Tasks
1. Automatically detect the content location. If it's challenging to detect, provide support for manual adjustment to square the location.
2. Finetune manga-ocr wiht Chinese or apply troce-chinese to this project.
3. Make the image as animate

## Reference source
1. https://zhuanlan.zhihu.com/p/661121944
2. https://github.com/kha-white/manga-ocr
3. https://github.com/chineseocr/trocr-chinese
4. https://github.com/juvian/Manga-Text-Segmentation
5. https://universe.roboflow.com/tarehimself/manga-translator-segmentation
6. https://github.com/reidenong/ComicPanelSegmentation
7. https://github.com/dmMaze/BallonsTranslator                   # handicraft the proper text blob
8. https://github.com/zyddnys/manga-image-translator/tree/main   # Prototype
9. DragDiffusion : https://blog.csdn.net/qq_44681809/article/details/135738479 