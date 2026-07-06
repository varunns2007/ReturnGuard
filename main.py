# ============================================================
#  ReturnGuard AI  —  Clothing Return Analysis System
#  Hackathon Edition v2.0
#  Run:  pip install flask pillow
#        python main.py
#  Open: http://127.0.0.1:5000
# ============================================================

from flask import Flask, render_template, request, jsonify
import os, base64, io, uuid, secrets, colorsys
from datetime import datetime, timedelta
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory session store
return_sessions = {}

# ─────────────────────────────────────────────────────────────
#  CLOTHING SUB-CATEGORY RISK TABLE
# ─────────────────────────────────────────────────────────────
CLOTHING_CATEGORY_RISK = {
    "Saree":       45,
    "Shirt":       40,
    "T-Shirt":     38,
    "Kurta":       42,
    "Jeans":       43,
    "Dress":       48,
    "Leggings":    35,
    "Jacket":      55,
    "Other":       40,
}

# Colour names for human-readable output
COLOUR_NAMES = {
    "red":    (255, 0,   0),
    "green":  (0,   128, 0),
    "blue":   (0,   0,   255),
    "white":  (255, 255, 255),
    "black":  (0,   0,   0),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0,   128),
    "pink":   (255, 192, 203),
    "brown":  (165, 42,  42),
    "grey":   (128, 128, 128),
    "cyan":   (0,   255, 255),
}

TAMIL = {
    "Return Approved": "திரும்ப பெறப்பட்டது ஒப்புதல்",
    "Manual Review":   "கையேடு மதிப்பாய்வு தேவை",
    "Fraud Suspected": "மோசடி சந்தேகிக்கப்படுகிறது",
}

CLOTHING_RETURN_REASONS = [
    "Wrong size delivered",
    "Color different from website",
    "Fabric quality issue",
    "Stitching defect",
    "Received wrong item",
    "Item damaged",
    "Other",
]


# ─────────────────────────────────────────────────────────────
#  UTILITY HELPERS
# ─────────────────────────────────────────────────────────────

def decode_image(data_url: str) -> Image.Image:
    raw = base64.b64decode(data_url.split(',')[1])
    return Image.open(io.BytesIO(raw)).convert('RGB')


def image_to_b64(img: Image.Image, fmt: str = 'PNG') -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def generate_otp(length: int = 6) -> str:
    import random, string
    return ''.join(random.choices(string.digits, k=length))


def rgb_to_hsv(r, g, b):
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def closest_colour_name(r, g, b) -> str:
    min_dist  = float('inf')
    best_name = "unknown"
    for name, (cr, cg, cb) in COLOUR_NAMES.items():
        dist = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if dist < min_dist:
            min_dist  = dist
            best_name = name
    return best_name


# ─────────────────────────────────────────────────────────────
#  FEATURE A — extract_dominant_colours()
#  Quantizes image into N dominant colours using PIL.
#  Returns list of {"hex", "name", "percent"} dicts.
# ─────────────────────────────────────────────────────────────

def extract_dominant_colours(img: Image.Image, n: int = 5) -> list:
    """
    Resize to thumbnail, quantize to N colours, return sorted list.
    Each entry: { hex, name, percent }
    """
    thumb = img.copy()
    thumb.thumbnail((150, 150))
    quantized = thumb.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    palette_img = quantized.convert('RGB')

    # Count pixels per colour bucket
    px_data    = list(palette_img.getdata())
    total      = len(px_data)
    colour_map = {}
    for px in px_data:
        key = px
        colour_map[key] = colour_map.get(key, 0) + 1

    # Sort by frequency, take top N
    sorted_colours = sorted(colour_map.items(), key=lambda x: -x[1])[:n]

    result = []
    for (r, g, b), count in sorted_colours:
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        name     = closest_colour_name(r, g, b)
        pct      = round(count / total * 100, 1)
        result.append({"hex": hex_code, "name": name, "percent": pct})
    return result


# ─────────────────────────────────────────────────────────────
#  FEATURE B — calculate_colour_difference()
#  Compares dominant colours between delivery vs return images.
#
#  Logic:
#    1. Extract top-5 colours from each image
#    2. Match each return colour to closest delivery colour in RGB space
#    3. Compute per-channel delta and overall colour shift score 0-100
#    4. Flag significant colour swap (e.g., red → blue = fraud signal)
# ─────────────────────────────────────────────────────────────

def calculate_colour_difference(img1: Image.Image,
                                 img2: Image.Image) -> dict:
    """
    Parameters
    ----------
    img1 : delivery image
    img2 : return image

    Returns
    -------
    colour_shift_score  : float  0-100 (0=identical colours, 100=completely different)
    dominant_delivered  : list   top colours in delivery photo
    dominant_returned   : list   top colours in return photo
    colour_matches      : list   per-colour match detail
    colour_verdict      : str    "Same Colour" | "Slight Variation" | "Major Colour Mismatch"
    colour_fraud_flag   : bool   True if dominant colour completely changed
    """
    delivered = extract_dominant_colours(img1, n=5)
    returned  = extract_dominant_colours(img2, n=5)

    matches   = []
    shift_sum = 0.0

    for ret_col in returned:
        ret_hex = ret_col['hex']
        rr = int(ret_hex[1:3], 16)
        rg = int(ret_hex[3:5], 16)
        rb = int(ret_hex[5:7], 16)

        best_dist  = float('inf')
        best_match = None

        for del_col in delivered:
            del_hex = del_col['hex']
            dr = int(del_hex[1:3], 16)
            dg = int(del_hex[3:5], 16)
            db = int(del_hex[5:7], 16)

            dist = ((rr - dr) ** 2 + (rg - dg) ** 2 + (rb - db) ** 2) ** 0.5
            if dist < best_dist:
                best_dist  = dist
                best_match = del_col

        # Normalise distance to 0-100
        norm_dist = round(min(100.0, best_dist / 4.41), 2)  # 4.41 ≈ sqrt(3*255²)/100
        shift_sum += norm_dist * (ret_col['percent'] / 100.0)

        matches.append({
            "returned_colour":  ret_col,
            "closest_delivery": best_match,
            "distance":         round(best_dist, 2),
            "shift_score":      norm_dist,
        })

    colour_shift_score = round(min(100.0, shift_sum), 2)

    # Verdict
    if colour_shift_score < 15:
        colour_verdict = "Same Colour"
    elif colour_shift_score < 40:
        colour_verdict = "Slight Variation"
    else:
        colour_verdict = "Major Colour Mismatch"

    # Fraud flag: dominant colour name completely changed
    top_del_name = delivered[0]['name'] if delivered else ""
    top_ret_name = returned[0]['name']  if returned  else ""
    colour_fraud_flag = (top_del_name != top_ret_name) and (colour_shift_score > 35)

    return {
        "colour_shift_score":  colour_shift_score,
        "dominant_delivered":  delivered,
        "dominant_returned":   returned,
        "colour_matches":      matches,
        "colour_verdict":      colour_verdict,
        "colour_fraud_flag":   colour_fraud_flag,
        "top_delivered_colour": top_del_name,
        "top_returned_colour":  top_ret_name,
    }


# ─────────────────────────────────────────────────────────────
#  FEATURE 1 — calculate_similarity()  (unchanged logic, clothing-tuned)
# ─────────────────────────────────────────────────────────────

def calculate_similarity(img1: Image.Image, img2: Image.Image) -> dict:
    SIZE = (480, 360)
    a = img1.resize(SIZE, Image.LANCZOS)
    b = img2.resize(SIZE, Image.LANCZOS)

    diff      = ImageChops.difference(a, b)
    diff_data = list(diff.getdata())
    total_px  = len(diff_data)

    mean_diff  = sum(r + g + bl for r, g, bl in diff_data) / (total_px * 3)
    similarity = max(0.0, 100.0 - (mean_diff / 255.0) * 100.0)

    minor_px    = sum(1 for r, g, bl in diff_data if (r + g + bl) / 3 > 10)
    moderate_px = sum(1 for r, g, bl in diff_data if (r + g + bl) / 3 > 40)
    severe_px   = sum(1 for r, g, bl in diff_data if (r + g + bl) / 3 > 80)

    minor_pct    = round(minor_px    / total_px * 100, 2)
    moderate_pct = round(moderate_px / total_px * 100, 2)
    severe_pct   = round(severe_px   / total_px * 100, 2)

    def avg_brightness(img):
        px = list(img.getdata())
        return sum((r + g + bl) / 3 for r, g, bl in px) / len(px)

    brightness_delta = round(avg_brightness(b) - avg_brightness(a), 2)

    damage_score = min(100.0, (
        minor_pct    * 0.3  +
        moderate_pct * 0.5  +
        severe_pct   * 2.5  +
        abs(brightness_delta) * 0.1
    ))

    diff_vis = diff.filter(ImageFilter.GaussianBlur(1))
    diff_vis = ImageEnhance.Brightness(diff_vis).enhance(5.0)
    diff_vis = ImageEnhance.Color(diff_vis).enhance(3.0)

    mismatch_percent = round(100.0 - similarity, 2)

    return {
        "similarity_percent":  round(similarity, 2),
        "mismatch_percent":    mismatch_percent,
        "minor_change_pct":    minor_pct,
        "moderate_change_pct": moderate_pct,
        "severe_change_pct":   severe_pct,
        "brightness_delta":    brightness_delta,
        "damage_score":        round(damage_score, 2),
        "diff_image":          f"data:image/png;base64,{image_to_b64(diff_vis)}",
    }


# ─────────────────────────────────────────────────────────────
#  FEATURE 2 — calculate_risk_score()
#  Now includes colour_shift_score as a 4th factor.
#
#  Formula:
#    risk_score = average(mismatch_pct, user_risk, category_risk, colour_shift)
#
#  0–40  →  Low
#  40–70 →  Medium
#  70–100→  High
# ─────────────────────────────────────────────────────────────

def calculate_risk_score(mismatch_pct:      float,
                          user_risk_score:   float,
                          sub_category:      str,
                          colour_shift_score: float) -> dict:
    """
    Parameters
    ----------
    mismatch_pct        : pixel-level mismatch 0-100
    user_risk_score     : behaviour score 0-100
    sub_category        : clothing sub-type string
    colour_shift_score  : colour difference score 0-100

    Returns
    -------
    risk_score     : float  0-100
    risk_level     : str    "Low" | "Medium" | "High"
    category_risk  : int
    """
    category_risk = CLOTHING_CATEGORY_RISK.get(sub_category,
                     CLOTHING_CATEGORY_RISK["Other"])

    raw_score  = (mismatch_pct + user_risk_score +
                  category_risk + colour_shift_score) / 4.0
    risk_score = round(min(100.0, max(0.0, raw_score)), 2)

    if risk_score < 40:
        risk_level = "Low"
    elif risk_score < 70:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "risk_score":    risk_score,
        "risk_level":    risk_level,
        "category_risk": category_risk,
    }


# ─────────────────────────────────────────────────────────────
#  FEATURE 3 — analyze_user_behavior()  (unchanged)
# ─────────────────────────────────────────────────────────────

def analyze_user_behavior(number_of_returns: int,
                           total_orders:      int,
                           account_age_days:  int) -> dict:
    if total_orders <= 0:
        total_orders = 1

    return_rate = (number_of_returns / total_orders) * 100.0
    flags = []

    if return_rate > 50:
        user_score = 80.0
        flags.append(f"High return frequency: {round(return_rate, 1)}% of orders returned")
    elif return_rate > 30:
        user_score = 60.0
        flags.append(f"Moderate return frequency: {round(return_rate, 1)}% of orders returned")
    else:
        user_score = 25.0

    if account_age_days < 30:
        user_score = min(100.0, user_score + 20.0)
        flags.append(f"New account: only {account_age_days} day(s) old — elevated scrutiny applied")

    return {
        "user_risk_score": round(user_score, 2),
        "return_rate":     round(return_rate, 2),
        "flags":           flags,
    }


# ─────────────────────────────────────────────────────────────
#  FEATURE 4 — get_decision()
#  For clothing: colour mismatch can override the similarity verdict.
#
#  Decision tree:
#    colour_fraud_flag = True          →  Fraud Suspected  (colour swap detected)
#    similarity > 85 AND colour OK     →  Return Approved
#    similarity > 85 AND colour varies →  Manual Review
#    60 ≤ similarity ≤ 85             →  Manual Review
#    similarity < 60                  →  Fraud Suspected
# ─────────────────────────────────────────────────────────────

def get_decision(similarity: float, colour_data: dict) -> dict:
    """
    Parameters
    ----------
    similarity   : float  0-100
    colour_data  : dict from calculate_colour_difference()

    Returns
    -------
    decision, decision_tamil, verdict_level, reason
    """
    colour_fraud_flag  = colour_data.get('colour_fraud_flag', False)
    colour_verdict     = colour_data.get('colour_verdict', '')
    colour_shift_score = colour_data.get('colour_shift_score', 0)

    # Colour swap overrides everything
    if colour_fraud_flag:
        decision      = "Fraud Suspected"
        verdict_level = "bad"
        reason        = (
            f"Colour swap detected: delivered '{colour_data['top_delivered_colour']}' "
            f"but returned '{colour_data['top_returned_colour']}'. "
            "This is a strong indicator of a wrong or substituted item."
        )

    elif similarity > 85 and colour_shift_score < 15:
        decision      = "Return Approved"
        verdict_level = "good"
        reason        = ("Product appears to be in original delivered condition "
                         "with matching colour — return approved.")

    elif similarity > 85 and colour_shift_score < 40:
        decision      = "Manual Review"
        verdict_level = "warn"
        reason        = ("Item looks similar but shows slight colour variation "
                         "(possible lighting difference). Team review needed.")

    elif similarity >= 60:
        decision      = "Manual Review"
        verdict_level = "warn"
        reason        = ("Moderate visual differences found. "
                         "A team member will verify the item before processing.")

    else:
        decision      = "Fraud Suspected"
        verdict_level = "bad"
        reason        = ("Significant visual mismatch detected. "
                         "Product may not match the original delivered item.")

    return {
        "decision":       decision,
        "decision_tamil": TAMIL.get(decision, ""),
        "verdict_level":  verdict_level,
        "reason":         reason,
    }


# ─────────────────────────────────────────────────────────────
#  FEATURE 5 — generate_explanation()  (now includes colour info)
# ─────────────────────────────────────────────────────────────

def generate_explanation(mismatch_pct:        float,
                          return_rate:         float,
                          sub_category:        str,
                          category_risk:       int,
                          risk_level:          str,
                          decision:            str,
                          user_flags:          list,
                          colour_data:         dict,
                          return_reason:       str) -> list:
    lines = []

    # 1. Pixel-level mismatch
    lines.append(
        f"Image mismatch: {round(mismatch_pct, 1)}% pixel difference "
        "between delivery and return photos"
    )

    # 2. Colour analysis
    lines.append(
        f"Colour analysis: {colour_data['colour_verdict']} "
        f"(shift score: {colour_data['colour_shift_score']}/100) — "
        f"Delivered dominant colour: '{colour_data['top_delivered_colour']}' | "
        f"Returned dominant colour: '{colour_data['top_returned_colour']}'"
    )

    if colour_data['colour_fraud_flag']:
        lines.append(
            "⚠️  COLOUR FRAUD FLAG: Dominant colour completely changed — "
            "returned item likely differs from delivered item."
        )

    # 3. User behaviour
    tag = "above normal" if return_rate > 30 else "within normal range"
    lines.append(
        f"User return rate: {round(return_rate, 1)}% ({tag})"
    )

    # 4. Clothing sub-category
    lines.append(
        f"Clothing type: {sub_category} — category risk score: {category_risk}/100"
    )

    # 5. Customer-stated reason
    lines.append(
        f"Customer stated return reason: '{return_reason}'"
    )

    # 6. User flags (high frequency, new account, etc.)
    lines.extend(user_flags)

    # 7. Final summary
    lines.append(
        f"Overall risk level: {risk_level} → Final decision: {decision}"
    )

    return lines


# ─────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/clothing/categories', methods=['GET'])
def get_clothing_categories():
    """Return available clothing sub-categories and return reasons."""
    return jsonify({
        "sub_categories":  list(CLOTHING_CATEGORY_RISK.keys()),
        "return_reasons":  CLOTHING_RETURN_REASONS,
    })


# STEP 1 — Delivery capture
@app.route('/delivery/capture', methods=['POST'])
def delivery_capture():
    data      = request.json
    image_b64 = data.get('image')
    order_id  = data.get('order_id') or ('CLO-' + uuid.uuid4().hex[:8].upper())

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    server_ts = datetime.now()
    ts_str    = server_ts.strftime('%Y-%m-%d %H:%M:%S')
    ts_file   = server_ts.strftime('%Y%m%d_%H%M%S')

    raw      = base64.b64decode(image_b64.split(',')[1])
    filename = f"delivery_{order_id}_{ts_file}.png"
    with open(os.path.join(UPLOAD_FOLDER, filename), 'wb') as f:
        f.write(raw)

    # Extract delivery-time dominant colours for later comparison
    delivery_img    = decode_image(image_b64)
    delivery_colours = extract_dominant_colours(delivery_img, n=5)

    session_id = uuid.uuid4().hex
    otp        = generate_otp()
    otp_expiry = server_ts + timedelta(days=30)

    return_sessions[session_id] = {
        "order_id":         order_id,
        "delivery_image":   image_b64,
        "delivery_ts":      ts_str,
        "delivery_file":    filename,
        "delivery_colours": delivery_colours,
        "otp":              otp,
        "otp_expires":      otp_expiry.isoformat(),
        "used":             False,
    }

    return jsonify({
        "success":          True,
        "order_id":         order_id,
        "session_id":       session_id,
        "delivery_ts":      ts_str,
        "return_otp":       otp,
        "delivery_colours": delivery_colours,  # useful for UI preview
    })


# STEP 2 — OTP validation
@app.route('/return/validate', methods=['POST'])
def validate_otp():
    data       = request.json
    session_id = data.get('session_id')
    otp        = data.get('otp')

    sess = return_sessions.get(session_id)
    if not sess:
        return jsonify({"valid": False, "error": "Invalid session ID"}), 400
    if sess['used']:
        return jsonify({"valid": False, "error": "Session already used"}), 400
    if datetime.now() > datetime.fromisoformat(sess['otp_expires']):
        return jsonify({"valid": False, "error": "OTP has expired"}), 400
    if sess['otp'] != otp:
        return jsonify({"valid": False, "error": "Incorrect OTP"}), 400

    return jsonify({
        "valid":       True,
        "order_id":    sess['order_id'],
        "delivery_ts": sess['delivery_ts'],
    })


# STEP 3 — Return capture + full analysis
@app.route('/return/capture', methods=['POST'])
def return_capture():
    data       = request.json
    session_id = data.get('session_id')
    otp        = data.get('otp')
    image_b64  = data.get('image')

    # Clothing-specific fields
    number_of_returns = int(data.get('number_of_returns', 2))
    total_orders      = int(data.get('total_orders',      10))
    account_age_days  = int(data.get('account_age_days',  90))
    sub_category      = data.get('sub_category', 'Other')   # e.g. "Shirt", "Saree"
    return_reason     = data.get('return_reason', 'Other')  # customer-stated reason

    # Session guard
    sess = return_sessions.get(session_id)
    if not sess:
        return jsonify({"error": "Invalid session"}), 400
    if sess['used']:
        return jsonify({"error": "Return session already used"}), 400
    if sess['otp'] != otp:
        return jsonify({"error": "OTP mismatch — fraud prevention triggered"}), 403
    if datetime.now() > datetime.fromisoformat(sess['otp_expires']):
        return jsonify({"error": "OTP expired"}), 400

    server_ts = datetime.now()
    ts_str    = server_ts.strftime('%Y-%m-%d %H:%M:%S')
    ts_file   = server_ts.strftime('%Y%m%d_%H%M%S')

    raw      = base64.b64decode(image_b64.split(',')[1])
    filename = f"return_{sess['order_id']}_{ts_file}.png"
    with open(os.path.join(UPLOAD_FOLDER, filename), 'wb') as f:
        f.write(raw)

    # ══════════════════════════════════════════════════════
    #  ANALYSIS PIPELINE
    # ══════════════════════════════════════════════════════

    img_delivery = decode_image(sess['delivery_image'])
    img_return   = decode_image(image_b64)

    # 1. Pixel-level similarity
    img_metrics = calculate_similarity(img_delivery, img_return)
    similarity  = img_metrics['similarity_percent']
    mismatch    = img_metrics['mismatch_percent']

    # 2. Colour difference analysis (NEW for clothing)
    colour_data = calculate_colour_difference(img_delivery, img_return)

    # 3. Decision (colour-aware)
    decision_data = get_decision(similarity, colour_data)

    # 4. User behaviour
    user_data = analyze_user_behavior(
                    number_of_returns, total_orders, account_age_days)

    # 5. Combined risk score (now includes colour shift)
    risk_data = calculate_risk_score(
                    mismatch,
                    user_data['user_risk_score'],
                    sub_category,
                    colour_data['colour_shift_score'])

    # 6. Explanation
    explanation = generate_explanation(
                    mismatch_pct        = mismatch,
                    return_rate         = user_data['return_rate'],
                    sub_category        = sub_category,
                    category_risk       = risk_data['category_risk'],
                    risk_level          = risk_data['risk_level'],
                    decision            = decision_data['decision'],
                    user_flags          = user_data['flags'],
                    colour_data         = colour_data,
                    return_reason       = return_reason,
                  )

    return_sessions[session_id]['used'] = True

    # ══════════════════════════════════════════════════════
    #  FINAL JSON RESPONSE
    # ══════════════════════════════════════════════════════
    return jsonify({
        "success":     True,
        "order_id":    sess['order_id'],
        "delivery_ts": sess['delivery_ts'],
        "return_ts":   ts_str,
        "analysis": {
            # Core
            "similarity":      similarity,
            "decision":        decision_data['decision'],
            "decision_tamil":  decision_data['decision_tamil'],
            "risk_score":      risk_data['risk_score'],
            "risk_level":      risk_data['risk_level'],
            "explanation":     explanation,
            "reason":          decision_data['reason'],
            "verdict_level":   decision_data['verdict_level'],

            # Pixel metrics
            "mismatch_percent":    mismatch,
            "damage_score":        img_metrics['damage_score'],
            "minor_change_pct":    img_metrics['minor_change_pct'],
            "moderate_change_pct": img_metrics['moderate_change_pct'],
            "severe_change_pct":   img_metrics['severe_change_pct'],
            "brightness_delta":    img_metrics['brightness_delta'],
            "diff_image":          img_metrics['diff_image'],

            # ── COLOUR ANALYSIS (new) ──────────────────────
            "colour_shift_score":  colour_data['colour_shift_score'],
            "colour_verdict":      colour_data['colour_verdict'],
            "colour_fraud_flag":   colour_data['colour_fraud_flag'],
            "dominant_delivered":  colour_data['dominant_delivered'],
            "dominant_returned":   colour_data['dominant_returned'],
            "colour_matches":      colour_data['colour_matches'],
            "top_delivered_colour": colour_data['top_delivered_colour'],
            "top_returned_colour":  colour_data['top_returned_colour'],

            # Risk breakdown
            "category_risk":   risk_data['category_risk'],
            "sub_category":    sub_category,
            "return_reason":   return_reason,
            "return_rate":     user_data['return_rate'],
            "user_risk_score": user_data['user_risk_score'],
        }
    })


if __name__ == '__main__':
    app.run(debug=True)