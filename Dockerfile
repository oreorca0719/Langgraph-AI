# 1. 베이스 이미지 설정
FROM python:3.11-slim

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. 필수 패키지 설치 및 환경 설정
# 도커 내에서 파이썬 출력을 실시간으로 확인하기 위해 설정
ENV PYTHONUNBUFFERED=1

# 4. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 프로젝트 전체 파일 복사 (중요!)
# 현재 디렉토리의 모든 파일과 폴더(app, static, templates, main.py 등)를 복사합니다.
COPY . .

# 6. 포트 노출
EXPOSE 8080

# 7. 실행 명령 (배포용 설정)
# --reload는 제외하고, worker 수를 조절하여 안정성을 높입니다.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]