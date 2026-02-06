@echo off
REM Генерация Python gRPC кода из proto файла
cd /d "%~dp0.."
python -m grpc_tools.protoc -I./proto --python_out=./src/recommendation_bert_api/generated --grpc_python_out=./src/recommendation_bert_api/generated ./proto/recommendation.proto
echo Done! Generated files in src/recommendation_bert_api/generated/
pause
