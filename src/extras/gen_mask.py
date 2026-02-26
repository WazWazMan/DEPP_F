from PIL import Image
from pathlib import Path

output_path = Path('./general_masks')
output_path.mkdir(exist_ok=True)

# columns
mask = Image.new("L", (512, 512), 0)
for i in range(512):
    for j in range(512):
        if i % 2 == 0:
            mask.putpixel((i, j), 255)

mask.save(output_path/"m_columns.png")


# rows
mask = Image.new("L", (512, 512), 0)
for i in range(512):
    for j in range(512):
        if j % 2 == 0:
            mask.putpixel((i, j), 255)

mask.save(output_path/"m_rows.png")


# dots
mask = Image.new("L", (512, 512), 0)
for i in range(512):
    for j in range(512):
        if (j+i) % 2 == 0:
            mask.putpixel((i, j), 255)

mask.save(output_path / f"m_dots.png")

# half
mask = Image.new("L", (512, 512), 0)
for i in range(512):
    for j in range(512):
        if i < (512/2):
            mask.putpixel((i, j), 255)

mask.save(output_path / f"m_half_0.png")

# half
mask = Image.new("L", (512, 512), 0)
for i in range(512):
    for j in range(512):
        if j < (512/2):
            mask.putpixel((i, j), 255)

mask.save(output_path / f"m_half_1.png")

# half
mask = Image.new("L", (512, 512), 0)
for i in range(512):
    for j in range(512):
        if i > (512/2):
            mask.putpixel((i, j), 255)

mask.save(output_path / f"m_half_2.png")

# half
mask = Image.new("L", (512, 512), 0)
for i in range(512):
    for j in range(512):
        if j > (512/2):
            mask.putpixel((i, j), 255)

mask.save(output_path / f"m_half_3.png")

# square_middle_inv
mask = Image.new("L", (512, 512), 0)
for i in range(512):
    for j in range(512):
        if i > 512/8*3 and j > 512/8*3 and i < 512/8*5 and j < 512/8*5:
            mask.putpixel((i, j), 255)

mask.save(output_path / f"m_square_middle_inv.png")

# square_middle
mask = Image.new("L", (512, 512), 255)
for i in range(512):
    for j in range(512):
        if i > 512/8*3 and j > 512/8*3 and i < 512/8*5 and j < 512/8*5:
            mask.putpixel((i, j), 0)

mask.save(output_path / f"m_square_middle.png")


mask.show()
