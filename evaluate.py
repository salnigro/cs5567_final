import os
import motmetrics as mm

def main():
    base_dir = r"c:\Users\salni\Documents\cs5567"
    gt_file = os.path.join(base_dir, "train", "MOT16-02", "gt", "gt.txt")
    ts_file = os.path.join(base_dir, "MOT16-02_results.txt")

    if not os.path.exists(gt_file):
        print(f"Ground truth file not found: {gt_file}")
        return
        
    if not os.path.exists(ts_file):
        print(f"Tracker results file not found: {ts_file}. Run tracker.py first.")
        return

    print("Loading Ground Truth and Tracker Output...")

    # Load data using motmetrics parser for MOT15/16 format:
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    # motmetrics expects ground truth to have confidence=1 or we can filter it.
    gt = mm.io.loadtxt(gt_file, fmt='mot15-2D', min_confidence=1)
    ts = mm.io.loadtxt(ts_file, fmt='mot15-2D')

    print("Comparing trajectories (this might take a few seconds)...")
    
    # Calculate distance matrix and create accumulator
    # We use 'iou' (Intersection over Union) as the distance metric with a threshold of 0.5
    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)

    mh = mm.metrics.create()
    
    # Compute the desired metrics
    summary = mh.compute(
        acc, 
        metrics=['num_frames', 'mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches'], 
        name='MOT16-02'
    )

    # Format the output table
    strsummary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap={
            'num_frames': 'Frames', 
            'mota': 'MOTA', 
            'motp': 'MOTP', 
            'idf1': 'IDF1', 
            'mostly_tracked': 'MT', 
            'mostly_lost': 'ML', 
            'num_false_positives': 'FP', 
            'num_misses': 'FN', 
            'num_switches': 'ID_SW'
        }
    )
    
    print("\nEvaluation Results:")
    print(strsummary)

if __name__ == "__main__":
    main()
