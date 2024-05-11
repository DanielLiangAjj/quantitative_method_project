## This is the lastest version of code after Professor Daniel's helpful feedbacks of:
## 1. Add multi-step forward prediction for both SFM and LSTM
## 2. Add relative mse when interpreting prediction result (we also added dataset variance for further clearification of the result)
## 3. Add plots of mse between predictions and actual data over time

## CNY/USD Prediction Result （Testing Dataset Variance: 0.026909392040267096):

**SFM** Prediction Result for **1 day** forward **Without Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.00043** and **relative mse of 0.016276**

![SFM | step 1 | no denoise | mse 0 00043 | relative mse 0 016276 | CNY](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/7d464f4f-b865-41a2-865e-e78d3124db9e)

![SFM_CNY_SINGLE_NO](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/885999b1-3698-462f-92f4-fbabf2f05f72)

**SFM** Prediction Result for **1 day** forward **Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.000335** and **relative mse of 0.012681**

![SFM | denoised | step 1 | mse 0 000335 | relative mse 0 012681 | CNY](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/217cb36f-99d4-4513-bfa5-1b90a009bc7f)

![SFM_CNY_SINGLE_DENOISE](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/2531e40a-2e5e-421b-a344-d2f38c8653c7)

**SFM** Prediction Result for **7 days** forward **Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.002535** and **relative mse of 0.107263**

![SFM | denoised | step 7 | mse 0 002535 | relative mse 0 107263 | CNY](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/ee67a385-8c8b-4c4d-82fd-75c41953f2f8)

![SFM_CNY_MULTI_DENOISE](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/7aaf2b83-7210-4489-aa25-442111ab7806)

**SFM** Prediction Result for **7 days** forward **Without Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.002984** and **relative mse of 0.126253**

![SFM | no denoise | step 7 | mse 0 002984 | relative mse 0 126253 | CNY](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/a2633696-f370-44b9-bef5-8a3887dfd2c1)

![SFM_CNY_MULTI_NO](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/7fcd5459-569c-4383-84d5-cc815da5767d)

**LSTM** Prediction Result for **1 day** forward **Without Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.0003530398104427771** and **relative mse of 0.0124182**

![WechatIMG448](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/df34f53b-2245-4c3d-9c8e-bb0fe38347d5)

![LSTM_CNY_SINGLE_MSE_NO_DENOISE](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/aed5b348-6769-4a5e-91cb-850b7c9ed1e0)

**LSTM** Prediction Result for **1 day** forward **Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.0002275145824628163** and **relative mse of 0.008454839**

![CNY_LSTM_with_wavelet](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/5f96bd89-ec29-496e-85ab-8bfb1288b21d)

![LSTM_CNY_SINGLE_MSE_DENOISE](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/49eaa828-b1d4-4ad6-a066-5f3336971d8e)

**LSTM** Prediction Result for **7 days** forward **Without Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.0037882266253429593** and **relative mse of 0.18334334425296941**

<img width="1015" alt="Screenshot 2024-05-08 at 6 32 29 AM" src="https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/6a01057a-b8ec-4d5a-9170-7b4ce5c7ec55">

![LSTM_CNY_MULTI_NO](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/b1ac409d-97b2-46a1-adb9-46e4418d9f73)

**LSTM** Prediction Result for **7 days** forward **Using Wavelet** Transform for **CNY/USD** Foreign Exchange Market with **mse of 0.004121296665305217** and **relative mse of 0.1994633341682037**

<img width="1014" alt="Screenshot 2024-05-08 at 6 35 58 AM" src="https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/03cf1881-997e-4455-bf69-2b614ca71301">

![LSTM_CNY_MULTI_DENOISED](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/6025c6a6-2ffe-41b1-a1fe-da04482da1ed)

## CAD/USD Prediction Result （Testing Dataset Variance: 0.00022684941404262516):
**Note:** Since the variance for CAD/USD dataset is very small compared to that of CNY/USD dataset, the mse and relative mse are different in the way that sometimes SFM might perform better and sometimes LSTM does, therefore making it difficult to draw a clear conclusion of which model is better under which scenario. Although the mse and relative mse from the CNY/USD dataset will also be different everytime the models are trained, the variance of the testing dataset is large enough such that, mse and relative mse of each model show consistent result that SFM is better at multi-step forward fx rate prediction, and LSTM is better at single day forward prediction.

**SFM** Prediction Result for **1 day** forward **Without Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.000022** and **relative mse of 0.08975**

![SFM | no_denoise | step 1 | mse 0 000022 | relative mse 0 08975 | CAD](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/6c4499c4-ed7a-4225-bccd-0c55a89d4919)

![SFM_CAD_SINGLE_NO](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/b53906bb-10ec-417f-8e55-43cb7f2b7e8f)

**SFM** Prediction Result for **1 day** forward **Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.000018** and **relative mse of 0.07289**

![SFM | denoised | step 1 | mse 0 000018 |  relative mse  0 07289 | CAD](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/eece816d-0526-4dff-8fc0-4ae4395f2db3)

![SFM_CAD_SINGLE_DENOISE](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/776a081a-bb8d-46c2-aece-10b683d43833)

**SFM** Prediction Result for **7 days** forward **Without Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.000159** and **relative mse of 0.708361**

![SFM | no_denoise | step 7 | mse 0 000159 | relative mse 0 708361 | CAD](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/921d1a76-b6c0-4f7c-b596-9e36390e7f8d)

![SFM_CAD_MSE_MULTI_NO](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/00adcbcc-c706-44bd-ae3b-52f71d764685)

**SFM** Prediction Result for **7 days** forward **Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.000147** and **relative mse of 0.65394**

![SFM | denoised | step 7 | mse 0 000147 | relative mse 0 65394 | CAD](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/abcf99b4-fd62-4132-914b-e42800220f4b)

![SFM_CAD_MULTI_DENOISE](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/bc285e2e-d201-49e7-8d93-34be5bbd8eaa)

**LSTM** Prediction Result for **1 day** forward **Without Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 5.980701870462418e-05** and **relative mse of 0.2477068888221327**

![cad_lstm_without_wavelet](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/07dc8003-4448-4eb4-8f65-927bd357960b)

![LSTM_CAD_SINGLE_NO](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/0d702e62-6fba-496b-b615-c7ed98e9eaf6)

**LSTM** Prediction Result for **1 day** forward **Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 2.3531084607891723e-05** and **relative mse of 0.09746032965827403**

![CAD_LSTM_with_wavelet](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/3cfae19b-6da0-473f-be86-707a7b27ab0e)

![LSTM_CAD_SINGLE_DENOISED](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/34685e80-0074-4ebf-865e-f7823380cee2)

**LSTM** Prediction Result for **7 days** forward **Without Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.00010998734393522309** and **relative mse of 0.4848473794803649**

<img width="990" alt="Screenshot 2024-05-08 at 5 50 24 AM" src="https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/30a3bb8d-3255-4906-a445-0b72c044433f">

![LSTM_CAD_MULTI_MSE_NO](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/36f41ab0-221c-482f-a1c4-6ea0ca52fdca)

**LSTM** Prediction Result for **7 days** forward **Using Wavelet** Transform for **CAD/USD** Foreign Exchange Market with **mse of 0.00012634266317769962** and **relative mse of 0.5569450717380287**

<img width="989" alt="Screenshot 2024-05-08 at 5 55 29 AM" src="https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/86828285-e0de-4906-b0bd-90964155a978">

![LSTM_CAD_MULTI_MSE_DENOISE](https://github.com/DanielLiangAjj/quantitative_method_project/assets/100398055/bf0b21ef-de69-49e5-affb-bd9c5f48e51c)

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

    
