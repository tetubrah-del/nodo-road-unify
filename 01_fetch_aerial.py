import requests
from PIL import Image
from io import BytesIO
import math

# 国土地理院 シームレス写真
TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def fetch_tile(z, x, y):
    url = TILE_URL.format(z=z, x=x, y=y)
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Missing tile: {z}/{x}/{y}")
        return None
    return Image.open(BytesIO(r.content))

def fetch_area_fixed(lat, lon, zoom=15, tiles=9):
    """
    lat, lon を中心に tiles x tiles 枚のタイルを取得して 1 枚に結合する。
    zoom=15, tiles=9 なら、おおよそ 12〜15km 四方カバーされます（辻原付近の緯度だと）。
    """
    cx, cy = deg2num(lat, lon, zoom)
    half = tiles // 2

    print(f"Center lat={lat}, lon={lon}, zoom={zoom}, tiles={tiles}x{tiles}")
    imgs = []

    for j in range(tiles):
        row = []
        for i in range(tiles):
            x = cx + (i - half)
            y = cy + (j - half)
            print(f"Fetch {zoom}/{x}/{y}")
            tile = fetch_tile(zoom, x, y)
            row.append(tile)
        imgs.append(row)

    # 最初に見つかったタイルのサイズを基準にキャンバスを作成
    base_tile = None
    for row in imgs:
        for t in row:
            if t is not None:
                base_tile = t
                break
        if base_tile is not None:
            break

    if base_tile is None:
        raise RuntimeError("全タイルが取得できませんでした")

    w, h = base_tile.size
    canvas = Image.new("RGB", (w * tiles, h * tiles))

    for j, row in enumerate(imgs):
        for i, img in enumerate(row):
            if img is not None:
                canvas.paste(img, (i * w, j * h))

    return canvas


if __name__ == "__main__":
    # 大分市 辻原 付近の代表座標
    lat = 33.14837
    lon = 131.5244

    # ★ ここを修正：zoom17 & tiles=3（= 約 1km 四方）
    img = fetch_area_fixed(lat, lon, zoom=17, tiles=3)

    img.save("aerial_tsujiwara_1km_z17.jpg")
    print("saved aerial_tsujiwara_1km_z17.jpg")

