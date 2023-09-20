# 해사데이터 기반 해양사고 위험요인 분석 (AIS, LTE-Maritime)

- 참여 인원 및 기간
    - 3명 (Data Engineer 담당)
    - 2022.07 ~ 2022.12
- 데이터: AIS, LTE-M 선박 항적 데이터 (위치 시계열 데이터)
- 과업 목표
    - 이종 시스템으로 수집되는 선박 항적 데이터 통계적 분석
    - 기존 시스템 대비 항적 예측 과업에 대한 신규 해상 무선통신 시스템의 적합성 실증
    - 차년도 항적 예측 과업에 활용 가능한 신규 데이터 특성 개발 (feature engineering)
- 사용 기술
    - Data Preprocessing: Python
    - Statistical Analysis: Stata
- 성과
    - 신규 전처리 방법론 : 항적간 거리 및 송수신 시간을 활용한 항적 항로화 방법론 개발, 이상치 제거 등
    - 신규 데이터 특성 개발: 항적간 거리 및 송수신 시간, 방위, 속도 등을 활용한 상호작용 변수 확인
    - 기존 AIS 수집 항적 데이터 대비 신규 LTE-Maritime 항적 데이터 활용 가치 입증
    - KCI 등재 학술지 연구 성과 게재
        - MIN, Ji Hong, et al. A Comparative Study of Vessel Trajectory Prediction Error based on AIS and LTE-Maritime Data. *Journal of Navigation and Port Research*, 2022, 46.6: 576-584.
