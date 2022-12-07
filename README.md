```c
import cv2
import mediapipe as mp
import numpy as np
import serial

ser = serial.Serial('COM8', 9600)
max_num_hands = 2
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition modelq
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)
ser.write(b' ')
while cap.isOpened():
    
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        rps_result = []

        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in rps_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': rps_gesture[idx],
                    'org': org
                })

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # Who wins?
            if len(rps_result) >= 2:
                winner = None
                text = ''
               
                

                if rps_result[0]['rps']=='rock':
                    if rps_result[1]['rps']=='rock'     : text = 'Tie'; ser.write(b' ');
                    elif rps_result[1]['rps']=='paper'  : text = 'Paper wins' ; ser.write(b'M');ser.write(b'L'); winner = 1
                    elif rps_result[1]['rps']=='scissors': text = 'Rock wins' ; ser.write(b'H');ser.write(b'N'); winner = 0
                elif rps_result[0]['rps']=='paper':
                    if rps_result[1]['rps']=='rock'     : text = 'Paper wins'  ; ser.write(b'H');ser.write(b'N');winner = 0
                    elif rps_result[1]['rps']=='paper'  : text = 'Tie'; ser.write(b' ');
                    elif rps_result[1]['rps']=='scissors': text = 'Scissors wins';ser.write(b'M');ser.write(b'L'); winner = 1
                elif rps_result[0]['rps']=='scissors':
                    if rps_result[1]['rps']=='rock'     : text = 'Rock wins'   ;ser.write(b'M');ser.write(b'L'); winner = 1
                    elif rps_result[1]['rps']=='paper'  : text = 'Scissors wins';ser.write(b'H');ser.write(b'N'); winner = 0
                    elif rps_result[1]['rps']=='scissors': text = 'Tie'; ser.write(b' ');

                if winner is not None:
                    cv2.putText(img, text='Winner', org=(rps_result[winner]['org'][0], rps_result[winner]['org'][1] + 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                cv2.putText(img, text=text, org=(int(img.shape[1] / 2), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
    
    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break
```    
    
# 마크다운

> * 이름 : 김진솔 
> * 학 번: 2217110191

>    > #### 학력
>    >  * 전안초등학교
>    >  * 광려중학교
>    >  * 마산중앙고등학교
>    >  * 한국폴리텍대학 창원캠퍼스 (스마트 팩토리과)

***

# AI-Control

### 내가 만든 첫 마크다운 파일입니다.

마크다운은 일반 텍스트 기반의 경량 마크업 언어다. 일반 텍스트로 서식이 있는 문서를 작성하는 데 사용되며, 일반 마크업 언어에 비해 문법이 쉽고 간단한 것이 특징이다. HTML과 리치 텍스트(RTF) 등 서식 문서로 쉽게 변환되기 때문에 응용 소프트웨어와 함께 배포되는 README 파일이나 온라인 게시물 등에 많이 사용된다.

장점: 간결하다.
 별도의 도구없이 작성가능하다.
 다양한 형태로 변환이 가능하다.
 텍스트(Text)로 저장되기 때문에 용량이 적어 보관이 용이하다.
 텍스트파일이기 때문에 버전관리시스템을 이용하여 변경이력을 관리할 수 있다.
 지원하는 프로그램과 플랫폼이 다양하다.
단점:  표준이 없다.
 표준이 없기 때문에 도구에 따라서 변환방식이나 생성물이 다르다.
 모든 HTML 마크업을 대신하지 못한다.

***
큰제목: 문서 제목

This is an H1
=============

작은제목: 문서 부제목

This is an H2
-------------

### This is a H3
#### This is a H4
##### This is a H5
###### This is a H6
####### This is a H7(지원하지 않음)

***
> 첫 번째 블록입니다.
>	> 두 번째 블록입니다.
>	>	> 세 번째 블록입니다.

1. 첫 번째
2. 두 번째
3. 세 번쨰

1. K
2. J
3. S


* 1단계
  - 2단계
    + 3단계
      + 4단계


  ***
 
  * 1개 
  * 1개 
  * 2개째
  * 2개째
  * ~~취소선~~
  
  ***
  
  
  * 참조링크

  [naver][naverlink]
  
  [naverlink]: https://naver.com "Go naver"
  
  ***
  
  - 외부링크
  
  [daum](https://www.daum.net)

***

* 자동연결
 
  외부링크: <http://example.com/>
  
  이메일링크: <address@example.com>
  
  ***
이미지 

  ![대체 텍스트(alternative text)를 입력하세요!](http://www.gstatic.com/webp/gallery/5.jpg "링크 설명(title)을 작성하세요.")


  
  
