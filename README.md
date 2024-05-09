## CNY/USD Prediction Result （Testing Dataset Variance: 0.026909392040267096):

**SFM** Prediction Result for **1 day** forward **Without Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.00043** and **relative mse of 0.016276**

![SFM | step 1 | no denoise | mse 0 00043 | relative mse 0 016276 | CNY](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/7d464f4f-b865-41a2-865e-e78d3124db9e)

**SFM** Prediction Result for **1 day** forward **Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.000335** and **relative mse of 0.012681**

![SFM | denoised | step 1 | mse 0 000335 | relative mse 0 012681 | CNY](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/217cb36f-99d4-4513-bfa5-1b90a009bc7f)

**SFM** Prediction Result for **7 days** forward **Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.002535** and **relative mse of 0.107263**

![SFM | denoised | step 7 | mse 0 002535 | relative mse 0 107263 | CNY](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/ee67a385-8c8b-4c4d-82fd-75c41953f2f8)

**SFM** Prediction Result for **7 days** forward **Without Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.002984** and **relative mse of 0.126253**

![SFM | no denoise | step 7 | mse 0 002984 | relative mse 0 126253 | CNY](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/a2633696-f370-44b9-bef5-8a3887dfd2c1)

**LSTM** Prediction Result for **1 day** forward **Without Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.0003530398104427771** and **relative mse of 0.0124182**

![WechatIMG448](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/df34f53b-2245-4c3d-9c8e-bb0fe38347d5)

**LSTM** Prediction Result for **1 day** forward **Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.0002275145824628163** and **relative mse of 0.008454839**

![CNY_LSTM_with_wavelet](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/5f96bd89-ec29-496e-85ab-8bfb1288b21d)

**LSTM** Prediction Result for **7 days** forward **Without Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.0037882266253429593** and **relative mse of 0.18334334425296941**

<img width="1015" alt="Screenshot 2024-05-08 at 6 32 29 AM" src="https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/6a01057a-b8ec-4d5a-9170-7b4ce5c7ec55">

**LSTM** Prediction Result for **7 days** forward **Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.004121296665305217** and **relative mse of 0.1994633341682037**

<img width="1014" alt="Screenshot 2024-05-08 at 6 35 58 AM" src="https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/03cf1881-997e-4455-bf69-2b614ca71301">


## CAD/USD Prediction Result （Testing Dataset Variance: 0.00022684941404262516):
**Note:** Since the variance for CAD/USD dataset is very small compared to that of CNY/USD dataset, the mse and relative mse are different in the way that sometimes SFM might perform better and sometimes LSTM does, therefore making it difficult to draw a clear conclusion of which model is better under which scenario. Although the mse and relative mse from the CNY/USD dataset will also be different everytime the models are trained, the variance of the testing dataset is large enough such that, mse and relative mse of each model show consistent result that SFM is better at multi-step forward fx rate prediction, and LSTM is better at single day forward prediction.

**SFM** Prediction Result for **1 day** forward **Without Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.000022** and **relative mse of 0.08975**

![SFM | no_denoise | step 1 | mse 0 000022 | relative mse 0 08975 | CAD](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/6c4499c4-ed7a-4225-bccd-0c55a89d4919)

**SFM** Prediction Result for **1 day** forward **Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.000018** and **relative mse of 0.07289**

![SFM | denoised | step 1 | mse 0 000018 |  relative mse  0 07289 | CAD](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/eece816d-0526-4dff-8fc0-4ae4395f2db3)

**SFM** Prediction Result for **7 days** forward **Without Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.000159** and **relative mse of 0.708361**

![SFM | no_denoise | step 7 | mse 0 000159 | relative mse 0 708361 | CAD](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/921d1a76-b6c0-4f7c-b596-9e36390e7f8d)

**SFM** Prediction Result for **7 days** forward **Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.000147** and **relative mse of 0.65394**

![SFM | denoised | step 7 | mse 0 000147 | relative mse 0 65394 | CAD](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/abcf99b4-fd62-4132-914b-e42800220f4b)

**LSTM** Prediction Result for **1 day** forward **Without Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 5.980701870462418e-05** and **relative mse of 0.2477068888221327**

![cad_lstm_without_wavelet](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/07dc8003-4448-4eb4-8f65-927bd357960b)

**LSTM** Prediction Result for **1 day** forward **Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 2.3531084607891723e-05** and **relative mse of 0.09746032965827403**

![CAD_LSTM_with_wavelet](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/3cfae19b-6da0-473f-be86-707a7b27ab0e)

**LSTM** Prediction Result for **7 days** forward **Without Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.00010998734393522309** and **relative mse of 0.4848473794803649**

<img width="990" alt="Screenshot 2024-05-08 at 5 50 24 AM" src="https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/30a3bb8d-3255-4906-a445-0b72c044433f">

**LSTM** Prediction Result for **7 days** forward **Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.00012634266317769962** and **relative mse of 0.5569450717380287**

<img width="989" alt="Screenshot 2024-05-08 at 5 55 29 AM" src="https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/86828285-e0de-4906-b0bd-90964155a978">


## Note
Relative MSE = MSE/VAR(y)

SFM stands for State Frequency Memory

LSTM stands for Long Short-Term Memory

This project uses three models: SFM, LSTM, and Multi-Step LSTM

## Project Inspired by 
Liheng Zhang, Charu Aggarwal, Guo-Jun Qi, Stock Price Prediction via Discovering Multi-Frequency Trading Patterns,
    in Proceedings of ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2017), Halifax, Nova Scotia,
    Canada, August 13-17, 2017.

## SFM Code Adapted and Modified from
[State_Frequency_Memory_Pytorch](https://github.com/yakouyang/State_Frequency_Memory_Pytorch)

    
