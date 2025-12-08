import json

def convert_clips_to_frames(input_file, output_file, frames_per_clip=30, total_frames=None):
    """
    Convert video clip classifications to frame-based format.
    
    Args:
        input_file: Path to input JSON file with clip classifications
        output_file: Path to output JSON file
        frames_per_clip: Number of frames in each clip (default: 30)
        total_frames: Total number of actual frames extracted (optional, for accurate end frame)
    """
    
    # Load input data
    with open(input_file, 'r') as f:
        clips = json.load(f)
    
    # Initialize output structure with proper capitalization
    output = {
        "Sitting": [],
        "Standing still": [],
        "Walking": [],
        "Upstair": [],
        "Downstair": []
    }
    
    # Map input actions to output keys
    action_map = {
        "sitting": "Sitting",
        "standing": "Standing still",
        "walking": "Walking",
        "upstair": "Upstair",
        "downstair": "Downstair"
    }
    
    # Group consecutive clips with the same action
    if not clips:
        return output
    
    current_action = clips[0]["classified_action"]
    start_frame = 1  # Start from frame 1
    
    for i, clip in enumerate(clips):
        action = clip["classified_action"]
        clip_start = i * frames_per_clip + 1
        clip_end = (i + 1) * frames_per_clip
        
        # If action changes or it's the last clip
        if action != current_action:
            # Save the previous action range
            mapped_action = action_map.get(current_action.lower(), current_action)
            output[mapped_action].append({
                "Starting frame": start_frame,
                "Ending frame": clip_start - 1
            })
            
            # Start new range
            current_action = action
            start_frame = clip_start
        
        # Handle last clip
        if i == len(clips) - 1:
            mapped_action = action_map.get(current_action.lower(), current_action)
            # Use actual total frames if provided, otherwise calculate
            final_frame = total_frames if total_frames else clip_end
            output[mapped_action].append({
                "Starting frame": start_frame,
                "Ending frame": final_frame
            })
    
    # Write output
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Conversion complete! Output saved to {output_file}")
    print(f"Total frames: {total_frames if total_frames else 'calculated'}")
    print(f"\nSummary:")
    for action, ranges in output.items():
        if ranges:
            total_action_frames = sum(r["Ending frame"] - r["Starting frame"] + 1 for r in ranges)
            print(f"  {action}: {len(ranges)} segment(s), {total_action_frames} frames")


# Example usage
if __name__ == "__main__":
    # Specify the actual total frames extracted (2331 in your case)
    convert_clips_to_frames('parkinson_proj/evaluation/evaluation_results/zero_shot_7b_results.json', 'translated_results.json', frames_per_clip=30, total_frames=2331)
    
    # Or let it calculate automatically:
    # convert_clips_to_frames('input.json', 'output.json', frames_per_clip=30)