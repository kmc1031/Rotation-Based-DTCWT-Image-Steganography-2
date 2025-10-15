import numpy as np
import pywt
from PIL import Image
import argparse


def text_to_bits(text):
    """텍스트를 비트 문자열로 변환"""
    bits = "".join(format(ord(c), "08b") for c in text)
    return bits


def bits_to_text(bits):
    """비트 문자열을 텍스트로 변환"""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i : i + 8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return "".join(chars)


def embed_message(image_path, message, output_path, wavelet="haar", strength=10):
    """
    DWT를 사용하여 이미지에 메시지 삽입

    Parameters:
    - image_path: 원본 이미지 경로
    - message: 삽입할 텍스트 메시지
    - output_path: 결과 이미지 저장 경로
    - wavelet: 사용할 웨이블릿 타입 (기본값: 'haar')
    - strength: 메시지 삽입 강도 (기본값: 10)
    """
    # 이미지 로드
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float64)

    # 메시지를 비트로 변환하고 종료 마커 추가
    message_with_delimiter = message + "###END###"
    bits = text_to_bits(message_with_delimiter)
    message_length = len(bits)

    # 메시지 길이를 32비트로 인코딩 (첫 부분에 저장)
    length_bits = format(message_length, "032b")
    total_bits = length_bits + bits

    # Blue 채널에 대해 DWT 수행
    blue_channel = img_array[:, :, 2]

    # 2D DWT 변환 수행
    coeffs = pywt.dwt2(blue_channel, wavelet)
    cA, (cH, cV, cD) = coeffs

    # cA (근사 계수)에 메시지 삽입
    cA_flat = cA.flatten().copy()

    # 메시지를 삽입할 공간이 충분한지 확인
    if len(total_bits) > len(cA_flat):
        raise ValueError(
            f"메시지가 너무 깁니다. 최대 {(len(cA_flat) - 32) // 8} 문자까지 가능합니다."
        )

    # 메시지 비트를 cA 계수에 삽입 (양자화 방식)
    for i, bit in enumerate(total_bits):
        coeff = cA_flat[i]
        # 계수를 양자화하여 비트 삽입
        quantized = int(coeff / strength) * strength
        if bit == "1":
            cA_flat[i] = quantized + strength * 0.75
        else:
            cA_flat[i] = quantized + strength * 0.25

    # 수정된 cA를 원래 형태로 복원
    cA_modified = cA_flat.reshape(cA.shape)

    # 역 DWT 수행
    coeffs_modified = (cA_modified, (cH, cV, cD))
    blue_channel_modified = pywt.idwt2(coeffs_modified, wavelet)

    # 원본 크기 유지
    h, w = blue_channel.shape
    blue_channel_modified = blue_channel_modified[:h, :w]

    # 수정된 Blue 채널을 이미지에 적용
    img_array[:, :, 2] = blue_channel_modified

    # 픽셀 값을 0-255 범위로 클리핑
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # 결과 이미지 저장 (PNG로 무손실 저장)
    result_img = Image.fromarray(img_array)
    result_img.save(output_path, "PNG")

    print(f"메시지가 성공적으로 삽입되었습니다: {output_path}")
    print(f"삽입된 메시지 길이: {len(message)} 문자")
    print(f"사용된 계수: {len(total_bits)} / {len(cA_flat)}")


def extract_message(image_path, wavelet="haar", strength=10):
    """
    DWT를 사용하여 이미지에서 메시지 추출

    Parameters:
    - image_path: 메시지가 포함된 이미지 경로
    - wavelet: 사용할 웨이블릿 타입 (기본값: 'haar')
    - strength: 메시지 삽입 시 사용한 강도

    Returns:
    - 추출된 텍스트 메시지
    """
    # 이미지 로드
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float64)

    # Blue 채널에서 DWT 수행
    blue_channel = img_array[:, :, 2]

    # 2D DWT 변환 수행
    coeffs = pywt.dwt2(blue_channel, wavelet)
    cA, (cH, cV, cD) = coeffs

    # cA에서 비트 추출
    cA_flat = cA.flatten()

    # 먼저 메시지 길이 추출 (첫 32비트)
    length_bits = ""
    for i in range(32):
        coeff = cA_flat[i]
        quantized = int(coeff / strength) * strength
        remainder = coeff - quantized
        # 0.75 strength에 가까우면 1, 0.25 strength에 가까우면 0
        if remainder > strength * 0.5:
            length_bits += "1"
        else:
            length_bits += "0"

    message_length = int(length_bits, 2)

    # 메시지 길이가 비정상적으로 크면 오류
    if message_length > len(cA_flat) * 8 or message_length < 0:
        return "오류: 메시지를 찾을 수 없습니다."

    # 실제 메시지 비트 추출
    bits = ""
    for i in range(32, 32 + message_length):
        if i >= len(cA_flat):
            break
        coeff = cA_flat[i]
        quantized = int(coeff / strength) * strength
        remainder = coeff - quantized
        if remainder > strength * 0.5:
            bits += "1"
        else:
            bits += "0"

    # 비트를 텍스트로 변환
    message = bits_to_text(bits)

    # 종료 마커 찾기
    delimiter_pos = message.find("###END###")
    if delimiter_pos != -1:
        message = message[:delimiter_pos]

    return message


def main():
    parser = argparse.ArgumentParser(description="DWT 기반 스테가노그래피")
    parser.add_argument(
        "mode",
        choices=["embed", "extract"],
        help="작동 모드: embed (삽입) 또는 extract (추출)",
    )
    parser.add_argument("--image", required=True, help="이미지 파일 경로")
    parser.add_argument("--message", help="삽입할 메시지 (embed 모드에서 필요)")
    parser.add_argument("--output", help="출력 이미지 경로 (embed 모드에서 필요)")
    parser.add_argument(
        "--wavelet", default="haar", help="웨이블릿 타입 (기본값: haar)"
    )
    parser.add_argument(
        "--strength", type=int, default=10, help="삽입 강도 (기본값: 10, 범위: 5-50)"
    )

    args = parser.parse_args()

    if args.mode == "embed":
        if not args.message or not args.output:
            print("오류: embed 모드에서는 --message와 --output이 필요합니다.")
            return
        embed_message(
            args.image, args.message, args.output, args.wavelet, args.strength
        )

    elif args.mode == "extract":
        message = extract_message(args.image, args.wavelet, args.strength)
        print(f"추출된 메시지: {message}")


if __name__ == "__main__":
    # 명령줄 인수가 없으면 예제 실행
    import sys

    if len(sys.argv) == 1:
        print("=" * 60)
        print("DWT 기반 스테가노그래피 프로그램")
        print("=" * 60)
        print("\n사용 예시:")
        print("\n1. 메시지 삽입:")
        print(
            "   python script.py embed --image input.png --message 'Hello World!' --output output.png"
        )
        print("\n2. 메시지 추출:")
        print("   python script.py extract --image output.png")
        print("\n3. 강도 조절 (선택사항):")
        print(
            "   python script.py embed --image input.png --message 'Secret' --output output.png --strength 20"
        )
        print("   python script.py extract --image output.png --strength 20")
        print("\n필요한 라이브러리:")
        print("   pip install numpy pillow pywavelets")
        print("\n참고:")
        print("   - 반드시 PNG 형식으로 저장하세요 (무손실)")
        print("   - strength 값이 클수록 강건하지만 화질 저하 가능")
        print("   - 삽입과 추출 시 동일한 strength 값 사용 필요")
        print("=" * 60)
    else:
        main()
