# Hướng dẫn cấu hình YouTube Data API v3

## Bước 1: Tạo Google Cloud Project

1. Truy cập [Google Cloud Console](https://console.cloud.google.com/)
2. Tạo project mới hoặc chọn project có sẵn
3. Đặt tên project và nhấn "Create"

## Bước 2: Bật YouTube Data API v3

1. Vào mục **APIs & Services** → **Library**
2. Tìm kiếm "YouTube Data API v3"
3. Nhấn vào kết quả và nhấn **Enable**

## Bước 3: Tạo API Key

1. Vào mục **APIs & Services** → **Credentials**
2. Nhấn **Create Credentials** → **API Key**
3. Copy API key được tạo ra

## Bước 4: Cấu hình API Key trong project

### Cách 1: Sử dụng Environment Variable (Khuyến nghị)

Tạo file `.env` trong thư mục gốc của project:

```bash
YOUTUBE_API_KEY=your_api_key_here
```

### Cách 2: Cập nhật trực tiếp trong settings.py

Mở file `core/settings.py` và thay đổi:

```python
YOUTUBE_API_KEY = 'your_api_key_here'
```

### Cách 3: Nhập trực tiếp trên giao diện

1. Truy cập trang Dashboard
2. Nhập API key vào ô "YouTube API Key"
3. Nhấn "Cập nhật dữ liệu"

## Bước 5: Kiểm tra API Key

Sau khi cấu hình, bạn có thể:

1. Truy cập trang Dashboard để xem dữ liệu thực từ YouTube
2. Sử dụng tính năng dự đoán video với dữ liệu thực
3. Xem danh sách video trending thực tế

## Lưu ý quan trọng

- **Quota**: YouTube Data API có giới hạn 10,000 requests/ngày cho mỗi project
- **Bảo mật**: Không chia sẻ API key với người khác
- **Monitoring**: Theo dõi usage trong Google Cloud Console
- **Cost**: API miễn phí trong giới hạn quota

## Troubleshooting

### Lỗi "API key not valid"
- Kiểm tra lại API key đã copy đúng chưa
- Đảm bảo YouTube Data API v3 đã được enable

### Lỗi "Quota exceeded"
- Đợi đến ngày hôm sau hoặc tạo project mới
- Kiểm tra usage trong Google Cloud Console

### Không hiển thị dữ liệu
- Kiểm tra kết nối internet
- Đảm bảo API key có quyền truy cập YouTube Data API v3

## Các API Endpoints được sử dụng

1. **Videos**: Lấy thông tin video trending
2. **Channels**: Lấy thông tin kênh
3. **VideoCategories**: Lấy tên thể loại video
4. **Search**: Tìm kiếm video (nếu cần)

## Monitoring và Analytics

- Theo dõi API usage trong Google Cloud Console
- Kiểm tra logs để debug lỗi
- Monitor performance của các API calls 