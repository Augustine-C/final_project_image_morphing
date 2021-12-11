# final_project

This project is contributed by [Yifan Cui](yifanc24@illinois.edu), [Zhenxi Li](zli89@illinois.edu) and [Anfeng Peng](anfengp2@illinois.edu).

In our final project, we explore several approaches for image morphing. We are initially motivated by the seamless morphing as seen in the “age transition” video played during the lecture. With some more research, we also see it has a wide range of applications in animation, film, font design, and medical imaging. We hope to gain a deeper understanding of the topic by implementing some image morphing techniques outlined in the lecture hands-on, including Cross-Dissolve, Align then cross-dissolve, and local warp, then cross-dissolve. Moreover, we think that implementing image morphing can fit well given the timeframe and scope of the final project. 


In this repository, there are 3 main jupyter notebooks: `Non_or_Aligned_Cross_Dissolve.ipynb`, `triangulation.ipynb` and `triangulation_with_auto_keypoint.ipynb`. 

`points.pkl` and `points_2.pkl` are saved key points from images. `utils.py` contains our main image-generatring functions. 

`shape_predictor_68_face_landmarks.dat` is from [italojs](https://github.com/italojs/facial-landmarks-recognition/blob/master/main.py)

Image used:
- [Cat image](https://icatcare.org/)
- [Tiger image](https://www.bbcearth.com/news/saving-the-amur-tiger)
- [Daniel Craig old&young image](https://www.tmz.com/2016/03/05/daniel-craig-good-genes-or-good-docs/)
- [River image and Bridge image](https://courses.engr.illinois.edu/cs445/fa2015/lectures/Lecture%2011%20-%20Image%20Morphing%20-%20CP%20Fall%202015.pdf)
- [Obama and micelle image]( 
https://www.goodreads.com/author/show/2338628.)
- [Michelle_Obama](https://www.irishtimes.com/news/social-affairs/religion-and-beliefs/thinking-anew-a-story-of-love-and-respect-1.4259535)
