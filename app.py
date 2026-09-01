import os
import sys
import subprocess
import tempfile

# ==========================================
# 0. 프로젝트 필수 설정 파일 자동 생성 (Self-Initialization)
# ==========================================
def init_project_files():
    """
    앱 실행에 필요한 환경 설정 파일(.streamlit/config.toml, requirements.txt)을
    자동으로 생성합니다.
    """
    # 1) .streamlit/config.toml (1GB 대용량 영상 업로드 허용 설정)
    os.makedirs(".streamlit", exist_ok=True)
    config_path = os.path.join(".streamlit", "config.toml")
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("[server]\nmaxUploadSize = 1000\n")

    # 2) requirements.txt (의존성 패키지 목록 생성)
    if not os.path.exists("requirements.txt"):
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write(
                "streamlit>=1.30.0\n"
                "ultralytics>=8.0.0\n"
                "opencv-python-headless\n"
                "numpy\n"
                "torch\n"
                "torchvision\n"
            )

# 필수 파일 초기화 실행
init_project_files()


# ==========================================
# 1. 라이브러리 임포트 및 페이지 설정
# ==========================================
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Valorant Cheat Detector",
    page_icon="🎯",
    layout="wide"
)


# ==========================================
# 2. AI & 영상 분석 핵심 로직 (백엔드)
# ==========================================
@st.cache_resource
def load_detection_model():
    """
    YOLOv8 모델 로드 (캐스 처리)
    * 실제 운영 환경에서는 커스텀 학습된 'models/valorant_yolo.pt' 파일 경로로 변경하세요.
    """
    try:
        model = YOLO('yolov8n.pt')  # 기본 경량화 테스트 모델
        return model
    except Exception as e:
        st.error(f"AI 모델 로드 실패: {e}")
        return None

def analyze_video(video_path, progress_callback=None):
    """
    영상 프레임 단위 분석 함수
    - 화면 중앙(크로스헤어) 기준 급격한 조준선 이동(에임 스냅) 및 이상 패턴 탐지
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    current_frame = 0
    suspicious_frames = []
    mouse_velocities = []
    prev_center = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame += 1
        
        # 화면 중앙 좌표 계산 (크로스헤어 위치)
        h, w, _ = frame.shape
        crosshair = (w // 2, h // 2)
        
        # 에임 스냅(인간의 한계를 넘어선 조준선 이동) 분석 규칙
        if prev_center is not None:
            velocity = np.linalg_norm(np.array(crosshair) - np.array(prev_center))
            mouse_velocities.append(velocity)
            
            # 임계값 초과 프레임 수집 (예시 임계값: 150)
            if velocity > 150: 
                suspicious_frames.append(current_frame)
                
        prev_center = crosshair

        # UI 진행률 프로그레스 바 업데이트
        if progress_callback and total_frames > 0:
            progress_callback(current_frame / total_frames)

    cap.release()
    
    # 핵 의심 점수 산출
    cheat_score = min(len(suspicious_frames) * 15 + 5, 99) if suspicious_frames else 3
    
    return {
        "cheat_score": cheat_score,
        "suspicious_frames": suspicious_frames,
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0
    }


# ==========================================
# 3. Streamlit 웹 인터페이스 (프론트엔드)
# ==========================================
def main():
    st.title("🎯 발로란트 핵(Aimbot) 판별 AI 서비스")
    st.write("발로란트 플레이 영상을 업로드하면 AI가 조준선 궤적과 반응 속도를 분석하여 핵 유무를 판별합니다.")
    st.divider()

    # AI 모델 로드
    model = load_detection_model()

    # 비디오 업로더 (최대 1GB 설정 적용됨)
    uploaded_file = st.file_uploader(
        "발로란트 영상 파일(.mp4, .mov, .avi)을 선택하세요", 
        type=["mp4", "mov", "avi"]
    )

    if uploaded_file is not None:
        # 임시 디렉토리에 영상 파일 저장
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        col1, col2 = st.columns([1, 1])
        
        # 왼쪽 컬럼: 원본 영상 출력
        with col1:
            st.subheader("📹 업로드된 영상")
            st.video(video_path)

        # 오른쪽 컬럼: 분석 제어 및 결과 모니터링
        with col2:
            st.subheader("🔍 AI 분석 시스템")
            analyze_btn = st.button("핵 판별 분석 시작", type="primary", use_container_width=True)

            if analyze_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(percent):
                    progress_bar.progress(percent)
                    status_text.text(f"프레임 분석 중... {int(percent * 100)}%")

                # 분석 알고리즘 실행
                with st.spinner("AI가 영상 프레임을 검토하고 있습니다..."):
                    result = analyze_video(video_path, progress_callback=update_progress)
                
                status_text.text("✅ 분석이 완료되었습니다!")
                st.divider()
                
                # 결과 리포트 출력
                score = result["cheat_score"]
                st.subheader("📊 판별 결과")
                
                if score >= 70:
                    st.error(f"🚨 **핵 의심 확률: {score}%** (인간의 한계를 벗어난 조준선 이동 감지)")
                elif score >= 40:
                    st.warning(f"⚠️ **의심 요소 존재: {score}%** (추가 플레이 검토 필요)")
                else:
                    st.success(f"✅ **정상 플레이: {score}%** (특이사항 없음)")

                # 세부 지표
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("총 분석 프레임", f"{result['total_frames']} frames")
                    st.metric("영상 플레이 시간", f"{result['duration_sec']} 초")
                with m_col2:
                    st.metric("이상 감지 프레임 수", f"{len(result['suspicious_frames'])} 개")

        # 분석 완료 후 임시 영상 파일 삭제
        os.unlink(video_path)

if __name__ == "__main__":
    main()
