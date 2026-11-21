from PIL import Image, ImageDraw

from gemma_ocr.config import Config
from gemma_ocr.render import crop_margins, fit, page_images, rotate, strips, upscale_small

BUDGET = Config().px_per_image  # 1120 * 48 * 48


def test_budget_is_gemma_max():
    assert BUDGET == 2_580_480


def test_fit_only_downscales():
    small = Image.new("RGB", (100, 100))
    assert fit(small, 10_000_000) is small
    big = fit(Image.new("RGB", (4000, 4000)), 1_000_000)
    assert big.width * big.height <= 1_000_000


def test_crop_margins_trims_white_border():
    img = Image.new("RGB", (1000, 1000), "white")
    ImageDraw.Draw(img).rectangle((200, 300, 800, 700), fill="black")
    c = crop_margins(img, pad=10)
    assert c.size == (621, 421)


def test_crop_margins_keeps_mostly_blank_page():
    img = Image.new("RGB", (1000, 1000), "white")
    ImageDraw.Draw(img).rectangle((10, 10, 60, 60), fill="black")
    assert crop_margins(img).size == (1000, 1000)


def test_strips_cover_page_with_overlap_at_native_res():
    img = Image.new("RGB", (2480, 3508))           # A4 @ 300 dpi
    ss = strips(img, BUDGET, 0.18, 6)
    assert 3 <= len(ss) <= 6
    assert all(s.width == 2480 for s in ss)
    assert all(s.width * s.height <= BUDGET for s in ss)
    assert ss[-1].height == ss[0].height            # last strip aligned to the bottom edge


def test_strips_none_when_page_fits_unless_forced():
    img = Image.new("RGB", (1500, 1500))
    assert strips(img, BUDGET) == []
    forced = strips(img, BUDGET, 0.18, 6, min_n=2)
    assert len(forced) == 2 and forced[0].height > 750


def test_strips_cap_at_max_n():
    assert len(strips(Image.new("RGB", (2480, 20000)), BUDGET, 0.18, 6)) == 6


def test_upscale_small_images_only():
    assert upscale_small(Image.new("RGB", (2000, 100)), 1700, 2600, 3.0).width == 2000
    assert upscale_small(Image.new("RGB", (1000, 100)), 1700, 2600, 3.0).width == 2600
    assert upscale_small(Image.new("RGB", (500, 100)), 1700, 2600, 3.0).width == 1500   # capped at 3x


def test_page_images_overview_first():
    cfg = Config()
    imgs, n = page_images(Image.new("RGB", (2480, 3508)), cfg)
    assert n == len(imgs) - 1 >= 3
    assert imgs[0].width * imgs[0].height <= BUDGET
    imgs, n = page_images(Image.new("RGB", (2480, 3508)), cfg, use_strips=False)
    assert n == 0 and len(imgs) == 1


def test_rotate():
    assert rotate(Image.new("RGB", (10, 20)), 90).size == (20, 10)
    assert rotate(Image.new("RGB", (10, 20)), 0).size == (10, 20)
