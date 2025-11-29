def calculate_danger_score(width_m, slope_deg, curvature, visibility, ground_condition):
    # simple weighted model
    return (
        (3 - width_m) * 0.4 +
        (slope_deg / 15) * 0.2 +
        (curvature / 50) * 0.2 +
        ((10 - visibility) / 10) * 0.1 +
        (ground_condition * 0.1)
    )
