import sys, cv2, os
sys.path.insert(0,'d:/project/TomatoVision/tomatovision-ml')
from tomato_detection_pipeline import TomatoDetectionPipeline

pipeline = TomatoDetectionPipeline()
out_dir  = 'd:/project/TomatoVision/tomatovision-ml/demo_outputs'
os.makedirs(out_dir, exist_ok=True)

tests = [
    ('C:/Users/Waruna/.gemini/antigravity/brain/d72fea25-187d-4266-99c6-341a7dceced8/.user_uploaded/media_1787157055647.png', '01_single_green_tomato'),
    ('C:/Users/Waruna/.gemini/antigravity/brain/d72fea25-187d-4266-99c6-341a7dceced8/.user_uploaded/media_1787156896370.png', '02_white_crate'),
    ('C:/Users/Waruna/.gemini/antigravity/brain/d72fea25-187d-4266-99c6-341a7dceced8/.user_uploaded/media_1787148501891.jpg', '03_user_tomatoes'),
    ('C:/Users/Waruna/.gemini/antigravity/brain/d72fea25-187d-4266-99c6-341a7dceced8/.user_uploaded/media_1787148462206.png', '04_tea_cup_zero'),
    ('C:/Users/Waruna/.gemini/antigravity/brain/d72fea25-187d-4266-99c6-341a7dceced8/.user_uploaded/media_1787159738196.jpg', '05_large_crate'),
    ('C:/Users/Waruna/.gemini/antigravity/brain/d72fea25-187d-4266-99c6-341a7dceced8/.user_uploaded/media_1787132875990.jpg', '06_heap'),
]

print('\n=== TomatoVision CNN Pipeline Demo Results ===')
for path, name in tests:
    r = pipeline.run_file(path)
    save_path = out_dir + '/' + name + '_detected.jpg'
    if r.annotated_image is not None:
        cv2.imwrite(save_path, r.annotated_image)
    ri = r.counts.get('ripe',0)
    un = r.counts.get('unripe',0)
    ov = r.counts.get('overripe',0)
    sp = r.counts.get('spoiled',0)
    print(name)
    print('  Scene  : ' + r.scene_mode + '  |  Total: ' + str(r.total_count) + '  [' + r.category + ']')
    print('  Counts : Ripe=' + str(ri) + ' Unripe=' + str(un) + ' Overripe=' + str(ov) + ' Spoiled=' + str(sp))
    print('  Quality: ' + str(r.quality_score) + '%  |  ' + r.grade)
    print()

print('Annotated outputs saved to: ' + out_dir)
