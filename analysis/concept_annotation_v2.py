"""
concept_annotation_v2.py
基于 CLIP embedding 的新 concept 数据
从 concept_report.csv 取数值，手动标注语义
运行后自动更新 visualize_concepts.py 和 decoder_clustering_v2.py
"""

# ── 新的 concept 标注（基于 CLIP embedding SAE 分析结果）────────
CONCEPTS = [
    # hate-biased features
    dict(id=3749, rate=0.025, bias=+0.025, type='hate',      concept='Anti-LGBTQ historical revisionism',  note='Alan Turing, JK Rowling, historical persecution framing'),
    dict(id=3784, rate=0.031, bias=+0.024, type='hate',      concept='Religious anti-LGBTQ rhetoric',      note='Christian framing, "faith vs. LGBT", persecution narrative'),
    dict(id=3763, rate=0.024, bias=+0.023, type='hate',      concept='Corporate/media anti-LGBTQ discourse',note='Disney, corporations targeted; "grooming kids" framing'),
    dict(id=3677, rate=0.025, bias=+0.023, type='hate',      concept='Political compass anti-trans memes',  note='Extremist political spectrum memes, anti-trans propaganda'),
    dict(id=670,  rate=0.020, bias=+0.023, type='hate',      concept='Ideological anti-LGBTQ framing',     note='Imperialist/reactionary framing, Alan Turing persecution'),
    dict(id=1023, rate=0.033, bias=+0.022, type='hate',      concept='Direct homophobic hostility',        note='"I hate gay people", gym harassment, anti-gay explicit content'),
    dict(id=1337, rate=0.030, bias=+0.022, type='hate',      concept='Anti-trans political rhetoric',      note='Trans kids, "transing children", conservative political memes'),
    dict(id=696,  rate=0.025, bias=+0.022, type='hate',      concept='Far-right anti-LGBTQ memes',         note='Declaration memes, flag controversy, nationalist framing'),
    dict(id=3043, rate=0.025, bias=+0.022, type='hate',      concept='Anti-trans propaganda templates',    note='Michael Knowles, political show screenshots, trans mockery'),
    dict(id=431,  rate=0.016, bias=+0.021, type='hate',      concept='Anti-LGBT foreign policy discourse', note='"LGBT as western imperialism", Japan/Russia framing'),

    # neutral features
    dict(id=2460, rate=0.035, bias=+0.008, type='neutral',   concept='LGBTQ legal/institutional discourse', note='Supreme Court, discrimination rulings, legal rights debate'),
    dict(id=2586, rate=0.034, bias=+0.003, type='neutral',   concept='Trans identity + mental health',      note='Trans boy icons, mental health activism, mixed framing'),
    dict(id=2446, rate=0.035, bias=-0.008, type='neutral',   concept='Gay slang / casual LGBTQ discourse',  note='"That\'s gay", coming out humor, casual identity references'),
    dict(id=2299, rate=0.036, bias=-0.008, type='neutral',   concept='Flag symbolism debate',               note='Pride flag, Confederate flag comparison, flag offense discourse'),
    dict(id=951,  rate=0.038, bias=+0.022, type='mild-hate', concept='Anti-pronoun / nofap rhetoric',       note='Neopronoun mockery, nofap anti-gay roots, fragile masculinity'),
    dict(id=170,  rate=0.035, bias=+0.021, type='hate',      concept='Anti-trans leftist propaganda framing',note='"LGBT as western propaganda", anti-trans CPAC content'),
    dict(id=3237, rate=0.037, bias=+0.025, type='hate',      concept='Political compass extremism',         note='Political compass memes, stonetoss, cold war gay rights'),

    # benign-biased features
    dict(id=1247, rate=0.038, bias=-0.045, type='benign',    concept='Trans rights affirmation',            note='Trans rights are human rights, trans masculine support'),
    dict(id=1897, rate=0.038, bias=-0.039, type='benign',    concept='LGBTQ+ celebrity/influencer support', note='Jacksepticeye, public figures affirming trans rights'),
    dict(id=1258, rate=0.038, bias=-0.033, type='benign',    concept='Trans community solidarity',          note='Trans telegram, community support, trans masculine icons'),
    dict(id=2961, rate=0.030, bias=-0.031, type='benign',    concept='Anti-transphobia activism',           note='"Me and the homies hate transphobes", protest content'),
    dict(id=1932, rate=0.032, bias=-0.031, type='benign',    concept='Coming out / identity disclosure',    note='Coming out stories, bisexual disclosure, family reactions'),
    dict(id=256,  rate=0.027, bias=-0.031, type='benign',    concept='LGBTQ+ humor / meme culture',         note='Gay humor, identity jokes, benign meme templates'),
    dict(id=3736, rate=0.040, bias=-0.031, type='benign',    concept='Trans masculine affirmation',         note='Trans masculine icons, DXRacer trans rights, community'),
    dict(id=1711, rate=0.034, bias=-0.030, type='benign',    concept='Pride event / protest coverage',      note='Anti-drag protests, activism coverage, pride defense'),
    dict(id=3234, rate=0.030, bias=-0.028, type='benign',    concept='Queer self-identification',           note='Gender questioning, "am I gay", identity exploration'),
]

# ── 输出为可直接 import 的 Python 格式 ────────────────────────
if __name__ == "__main__":
    import os
    import json

    OUT = "/projectnb/cepinet/users/Jay/InterVLP"

    # 保存为 JSON 供其他脚本读取
    out_path = f"{OUT}/sae_outputs/concepts_v2.json"
    with open(out_path, 'w') as f:
        json.dump(CONCEPTS, f, indent=2)
    print(f"✓ 保存到: {out_path}")

    # 打印摘要
    from collections import Counter
    type_counts = Counter(c['type'] for c in CONCEPTS)
    print(f"\n总计 {len(CONCEPTS)} 个 feature:")
    for t, n in type_counts.items():
        print(f"  {t:12s}: {n}")

    print(f"\nTop hate-biased:")
    for c in sorted([x for x in CONCEPTS if x['bias']>0], key=lambda x:-x['bias'])[:5]:
        print(f"  #{c['id']:4d}  bias={c['bias']:+.3f}  {c['concept']}")

    print(f"\nTop benign-biased:")
    for c in sorted([x for x in CONCEPTS if x['bias']<0], key=lambda x:x['bias'])[:5]:
        print(f"  #{c['id']:4d}  bias={c['bias']:+.3f}  {c['concept']}")
