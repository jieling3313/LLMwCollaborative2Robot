# LLMwCollaborative2Robot
# Prototype Design of a Voice-Driven Collaborative Robot System Using LLM
![image](https://github.com/user-attachments/assets/c36b18b0-e051-445e-bd65-31edc820a62e)
在LLM策略設計中，我們設計了以下三個重要文檔: Domain.txt、Example.txt以及Prompt.txt。
在Domain.txt中，為了使LLM更加規劃策略，使用PDDL(Planning Domain Definition Language)撰寫的領域定義文件。
    首先，文件在:types分別定義三個基本型別：robot(機器人)、location(位置)和item(物品)。
    接下來，在:predicates部分，文件描述了一組謂詞，用來表示不同實體之間的狀態與關係。
    在Action Space中，每個動作都以:action開始，並包含參數、前置條件(:precondition)和效應(:effect)三個部分。
    整體設計思路基於STRIPS規劃，幫助LLM生成滿足所有前置條件與效應要求的多機器人行動序列。
在Example.txt中於LLM容易生成“幻覺”輸出，出現如格式不一，或任務無相關輸出，所以需要藉由此格式，制定統一規範於此。
位置定位(location):
    在example.txt中定義了loc_x_y位置，代表了工作空間中各個具體的座標。
    而子位置(如:sub_x_y_1和sub_x_y_2)則用於表明不同機器人在同一主要位置下進行操作的子空間。文件分為兩種格式，使機器人更容易讀取低階操作。
功能(function):
    文件中詳細列出了每個time step中各機器人的動作，比如robot(1)在sub_5_5_1執行動作，robot2在sub_5_5_2執行動作；
    接著robot1(1)執行pick_high，而robot(2)則不進行操作；
    之後再輪到robot(2)執行pick_low，然後 robot(1)執行place_low，robot(2)執行place_high，最後(done)作為任務結束。
    利用這種時間序列式的動作排列，不僅能展示多機器人協同工作的動作規劃，更能體現了各個動作之間的依賴關係。
Prompt.txt中，利用頭頂相機捕捉ArUco標記的參數數據匯入電腦後，系統計算各實體的相對位置與狀態獲取環境資訊。
    使用者透過錄音並利用Whisper模型轉錄匯入語音指令。如「請幫我將boxA放到boxB上」。
