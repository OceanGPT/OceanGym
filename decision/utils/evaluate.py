import os
import sys
import json
import glob
import math
import re
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import statistics 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
from config_process import load_config
config = load_config()
base_path = config["global"]["base_path"]

# Add project root directory to sys.path


from utils.calculate import DistanceCalculator


def setup_logging(eval_root: str):
    """
    Setup logging to both console and file for each eval_root
    """
    # clear all handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_dir = os.path.join(eval_root, "eval")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"auv_evaluation_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging started. Log file: {log_filepath}")
    return logger, log_filepath



class AUVPerformanceEvaluator:
    """Evaluate AUV performance by analyzing memory files and calculating distances to target objects"""
    
    def __init__(self):         
        self.distance_calculator = DistanceCalculator()
        self.config = load_config()
        
        # Map task IDs to target object names
        self.task_target_mapping = {
            "task1": "MINING ROBOT",
            "task2": "OIL PIPELINE", 
            "task3": "OIL DRUM",
            "task4": "SUNKEN SHIP",
            "task5": "ELECTRICAL BOX",
            "task6": "WIND POWER STATION",
            "task7": "AIRCRAFT WRECKAGE",
            "task8": "H-MARKED LANDING PLATFORM",
            # Additional task mappings
            "task9": "OIL PIPELINE",      # task9 corresponds to task2 (oil pipeline)
            "task10": "OIL DRUM",         # task10 corresponds to task3 (oil drum)
            "task11": "SUNKEN SHIP",      # task11 corresponds to task4 (sunken ship)
            "task12": "WIND POWER STATION" # task12 corresponds to task6 (wind power)
        }
    
    def get_target_info_with_positions(self, target_name: str) -> Dict:
        """
        Get target information including all positions
        
        Args:
            target_name: Name of the target object
            
        Returns:
            Dict: Target information with positions
        """
        if target_name not in self.distance_calculator.target_coordinates:
            return {"error": f"Target '{target_name}' not found"}
        
        target_coords = self.distance_calculator.target_coordinates[target_name]
        
        return {
            "target_name": target_name,
            "target_count": len(target_coords),
            "target_positions": [
                {
                    "point_index": i + 1,
                    "coordinate": coord,
                    "position_string": f"({coord[0]}, {coord[1]}, {coord[2]})"
                }
                for i, coord in enumerate(target_coords)
            ]
        }
    
    def calculate_score_for_distance(self, distance: float) -> float:
        """
        Calculate score based on distance to target for a single point
        Updated scoring criteria: <=30m full score, 30-100m proportional decrease, >100m zero score
        
        Args:
            distance: Distance to target
            
        Returns:
            float: Calculated score (0-100)
        """
        # Distance-based scoring calculation for single point
        if distance <= 30:
            # Distance <= 30m, give full score 100
            return 100.0
        elif distance > 100:
            # Distance > 100m, give no score
            return 0.0
        else:
            # 30m < distance <= 100m, proportional decrease
            # Linear decrease from 30m (100 points) to 100m (0 points)
            distance_ratio = (100 - distance) / (100 - 30)  # Closer distance means higher ratio
            return 100.0 * distance_ratio

    def calculate_weighted_score_for_multiple_targets(self, distances: List[float], target_count: int) -> Dict:
        """
        Calculate weighted score for multiple target points based on new evaluation criteria
        
        Args:
            distances: List of distances to each target point
            target_count: Number of target points
            
        Returns:
            Dict: Calculated weighted score and breakdown details
        """
        if not distances or target_count == 0:
            return {
                "total_score": 0.0,
                "score_breakdown": [],
                "weights_used": []
            }
        
        # Sort distances to get the closest ones first
        sorted_distances = sorted(distances)
        
        # Define weight distribution based on target count
        if target_count == 1:
            # Single point: 100% weight
            weights = [100.0]
        elif target_count == 2:
            # Two points: 60% + 40%
            weights = [60.0, 40.0]
        elif target_count >= 3:
            # Three or more points: 60% + 20% + 20% (only use first 3)
            weights = [60.0, 20.0, 20.0]
        else:
            return {
                "total_score": 0.0,
                "score_breakdown": [],
                "weights_used": []
            }
        
        total_score = 0.0
        score_breakdown = []
        weights_used = []
        
        # Calculate weighted score for each distance
        for i, distance in enumerate(sorted_distances[:len(weights)]):
            # Get individual score for this distance
            individual_score = self.calculate_score_for_distance(distance)
            
            # Apply weight (weight is percentage, so divide by 100)
            weighted_score = (individual_score / 100.0) * weights[i]
            total_score += weighted_score
            
            score_breakdown.append({
                "distance": distance,
                "individual_score": individual_score,
                "weight_percentage": weights[i],
                "weighted_score": weighted_score
            })
            weights_used.append(weights[i])
        
        return {
            "total_score": total_score,
            "score_breakdown": score_breakdown,
            "weights_used": weights_used
        }

    def extract_timestamp_from_filename(self, filename: str) -> str:
        """
        Extract timestamp from filename
        
        Args:
            filename: File name like "important_memory_2025-09-09_15-26-57.json"
            
        Returns:
            str: Timestamp like "2025-09-09_15-26-57"
        """
        # Match timestamp pattern
        pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
        match = re.search(pattern, filename)
        return match.group(1) if match else None
    
    def group_files_by_timestamp(self, file_paths: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Group files by their timestamp
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dict: Grouped files by timestamp
                  {timestamp: {"memory": path, "important_memory": path}}
        """
        grouped_files = defaultdict(dict)
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            timestamp = self.extract_timestamp_from_filename(filename)
            
            if timestamp:
                if filename.startswith("important_memory"):
                    grouped_files[timestamp]["important_memory"] = file_path
                elif filename.startswith("memory"):
                    grouped_files[timestamp]["memory"] = file_path
        
        return dict(grouped_files)
    
    def extract_locations_from_memory_file(self, file_path: str) -> List[List[float]]:
        """
        Extract location data from memory file
        
        Args:
            file_path: Path to the memory JSON file
            
        Returns:
            List[List[float]]: List of location coordinates
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            locations = []
            for entry in data:
                if isinstance(entry, dict) and "location" in entry:
                    location = entry["location"]
                    if isinstance(location, list) and len(location) >= 3:
                        locations.append(location[:3])  # Take only x, y, z coordinates
            
            return locations
        except Exception as e:
            logging.error(f"❌ Error reading file {file_path}: {e}")
            return []
    
    def find_closest_distance_to_targets(self, locations: List[List[float]], 
                                       task_id: str, use_last_only: bool = False) -> Dict:
        """
        Find the closest distance to target objects with new weighted scoring
        For important_memory files: find the location closest to the first target point, then evaluate from that location
        For memory files: use the last location only
        
        Args:
            locations: List of AUV location coordinates
            task_id: Task ID (e.g., "task1", "task2")
            use_last_only: If True, only use the last location; if False, find closest to first target point
            
        Returns:
            Dict: Analysis results including closest distance and weighted score
        """
        if not locations:
            return {"error": "No locations provided"}
        
        # Get target name from task ID
        target_name = self.task_target_mapping.get(task_id)
        if not target_name:
            return {"error": f"Unknown task ID: {task_id}"}
        
        if target_name not in self.distance_calculator.target_coordinates:
            return {"error": f"Target '{target_name}' not found in target coordinates"}
        
        target_coords = self.distance_calculator.target_coordinates[target_name]
        target_count = len(target_coords)
        
        # Get target information with positions
        target_info = self.get_target_info_with_positions(target_name)
        
        # Determine which location to use for evaluation
        if use_last_only:
            # Memory files: Only use the last location
            selected_auv_location = locations[-1]
            selected_auv_index = len(locations) - 1
            strategy = "last_point"
        else:
            # Important_memory files: Find the location closest to the first target point
            first_target_coord = target_coords[0]  # Use the first target point as reference
            first_target_tuple = tuple(first_target_coord)
            
            min_distance_to_first_target = float('inf')
            selected_auv_location = None
            selected_auv_index = -1
            
            # Find the AUV location closest to the first target point
            for i, auv_location in enumerate(locations):
                auv_tuple = tuple(auv_location[:3])
                distance_to_first_target = self.distance_calculator.calculate_3d_distance(auv_tuple, first_target_tuple)
                
                if distance_to_first_target < min_distance_to_first_target:
                    min_distance_to_first_target = distance_to_first_target
                    selected_auv_location = auv_location
                    selected_auv_index = i
            
            strategy = "closest_to_first_target"
        
        # Now calculate distances from the selected AUV location to all target points
        selected_auv_tuple = tuple(selected_auv_location[:3])
        distances_to_all_targets = []
        overall_min_distance = float('inf')
        best_target_coordinate = None
        best_target_index = -1
        
        for target_idx, target_coord in enumerate(target_coords):
            target_tuple = tuple(target_coord)
            distance = self.distance_calculator.calculate_3d_distance(selected_auv_tuple, target_tuple)
            distances_to_all_targets.append(distance)
            
            # Track overall minimum distance for reference
            if distance < overall_min_distance:
                overall_min_distance = distance
                best_target_coordinate = target_coord
                best_target_index = target_idx
        
        # Calculate weighted score based on distances from selected location
        score_result = self.calculate_weighted_score_for_multiple_targets(distances_to_all_targets, target_count)
        weighted_score = score_result["total_score"]
        
        return {
            "task_id": task_id,
            "target_name": target_name,
            "target_info": target_info,
            "target_count": target_count,
            "minimum_distance": overall_min_distance if overall_min_distance != float('inf') else None,
            "distances_to_all_targets": distances_to_all_targets,
            "weighted_score": weighted_score,
            "score_breakdown": score_result,
            "closest_auv_location": selected_auv_location,
            "closest_target_coordinate": best_target_coordinate,
            "auv_location_index": selected_auv_index,
            "target_index": best_target_index,
            "use_last_only": use_last_only,
            "strategy": strategy,
            "total_locations": len(locations),
            "locations_analyzed": 1  # Always analyze only one selected location
        }

    def evaluate_single_file(self, file_path: str, task_id: str) -> Dict:
        """
        Evaluate a single memory file with weighted scoring
        
        Args:
            file_path: Path to the memory file
            task_id: Task ID (e.g., "task1")
            
        Returns:
            Dict: Evaluation results with weighted score
        """
        locations = self.extract_locations_from_memory_file(file_path)
        
        if not locations:
            return {
                "error": "No valid locations found in file",
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "weighted_score": 0,
                "strategy": "no_locations"
            }
        
        # Determine file type and strategy
        file_name = os.path.basename(file_path)
        is_important_memory = file_name.startswith("important_memory")
        use_last_only = not is_important_memory  # memory files use last only, important_memory finds closest to first target
        
        result = self.find_closest_distance_to_targets(locations, task_id, use_last_only)
        
        # Use weighted score as the primary score
        if "error" in result or result["minimum_distance"] is None:
            weighted_score = 0
        else:
            weighted_score = result.get("weighted_score", 0)
        
        # Check for recognition error (score < 30 but returned success)
        recognition_error = False
        if weighted_score < 30 and "error" not in result:
            recognition_error = True
        
        result.update({
            "file_path": file_path,
            "file_name": file_name,
            "file_type": "important_memory" if is_important_memory else "memory",
            "weighted_score": weighted_score,  # Primary weighted score
            # "recognition_error": recognition_error,
            "total_locations": len(locations)
        })
        
        return result

    def evaluate_file_group(self, file_group: Dict[str, str], task_id: str, timestamp: str) -> Dict:
        """
        Evaluate a group of files with the same timestamp using weighted scoring
        
        Args:
            file_group: Dictionary with "memory" and/or "important_memory" file paths
            task_id: Task ID
            timestamp: Timestamp of the file group
            
        Returns:
            Dict: Combined evaluation results for the file group
        """
        group_result = {
            "timestamp": timestamp,
            "task_id": task_id,
            "file_results": {},
            "best_result": None,
            "group_best_weighted_score": 0,
            "group_best_distance": None,
            "has_recognition_error": False
        }
        
        best_weighted_score = 0
        best_result = None
        has_recognition_error = False
        
        # Evaluate each file type in the group
        for file_type, file_path in file_group.items():
            if file_path and os.path.exists(file_path):
                file_result = self.evaluate_single_file(file_path, task_id)
                file_result["timestamp"] = timestamp
                group_result["file_results"][file_type] = file_result
                
                # Check for recognition errors
                if file_result.get("recognition_error", False):
                    has_recognition_error = True
                
                # Check if this is the best result in the group (highest weighted score)
                current_weighted_score = file_result.get("weighted_score", 0)
                if current_weighted_score > best_weighted_score:
                    best_weighted_score = current_weighted_score
                    best_result = file_result
        
        group_result["best_result"] = best_result
        group_result["group_best_weighted_score"] = best_weighted_score
        group_result["has_recognition_error"] = has_recognition_error
        if best_result and "minimum_distance" in best_result:
            group_result["group_best_distance"] = best_result["minimum_distance"]
        
        return group_result

    def calculate_task_average_score(self, all_file_groups: List[Dict]) -> Dict:
        """
        Calculate average score for a task based on all file groups
        
        Args:
            all_file_groups: List of all file group results for this task
            
        Returns:
            Dict: Average score statistics
        """
        weighted_scores = []
        total_groups = len(all_file_groups)
        recognition_errors = 0
        
        for group in all_file_groups:
            group_weighted_score = group.get("group_best_weighted_score", 0)
            
            # Include all scores, including 0 scores
            weighted_scores.append(group_weighted_score)
            
            # Count recognition errors
            if group.get("has_recognition_error", False):
                recognition_errors += 1
        
        if not weighted_scores:
            return {
                "average_weighted_score": 0,
                "valid_groups": 0,
                "total_groups": total_groups,
                "max_weighted_score": 0,
                "min_weighted_score": 0,
                "weighted_score_distribution": [],
                "recognition_errors": 0
            }
        
        average_weighted_score = sum(weighted_scores) / len(weighted_scores)
        
        return {
            "average_weighted_score": average_weighted_score,
            "valid_groups": len([s for s in weighted_scores if s > 0]),  # Number of valid scores (>0)
            "total_groups": total_groups,
            "max_weighted_score": max(weighted_scores),
            "min_weighted_score": min(weighted_scores),
            "weighted_score_distribution": sorted(weighted_scores, reverse=True),
            "recognition_errors": recognition_errors
        }
        
    def evaluate_task_directory(self, task_dir: str) -> Dict:
        """
        Evaluate all memory files in a task directory grouped by timestamp

        Args:
            task_dir: Path to task directory

        Returns:
            Dict: Combined evaluation results for the task
        """
        task_id = os.path.basename(task_dir)
        results = {
            "task_id": task_id,
            "task_directory": task_dir,
            "point_results": {},
            "overall_best": None,
            "all_file_groups": [],
            "overall_best_weighted_score": 0,
            "task_average_weighted_score": 0,
            "score_statistics": {}
        }

        # Find all point directories
        point_pattern = os.path.join(task_dir, "point*")
        point_dirs = glob.glob(point_pattern)

        overall_best_weighted_score = 0
        overall_best_result = None

        for point_dir in point_dirs:
            point_name = os.path.basename(point_dir)

            # Find both important_memory and memory files
            important_memory_pattern = os.path.join(point_dir, "important_memory_*.json")
            memory_pattern = os.path.join(point_dir, "memory_*.json")

            important_memory_files = glob.glob(important_memory_pattern)
            memory_files = glob.glob(memory_pattern)

            all_memory_files = important_memory_files + memory_files

            # Group files by timestamp
            grouped_files = self.group_files_by_timestamp(all_memory_files)

            point_results = {
                "point_name": point_name,
                "point_directory": point_dir,
                "file_groups": [],
                "best_result": None,
                "total_file_groups": len(grouped_files),
                "important_memory_files": len(important_memory_files),
                "memory_files": len(memory_files),
                "point_best_weighted_score": 0,
                "point_average_weighted_score": 0,
                "point_weighted_std": 0.0  # Use std instead of variance
            }

            best_weighted_score_in_point = 0
            best_result_in_point = None
            point_weighted_scores = []

            # Evaluate each file group
            for timestamp, file_group in grouped_files.items():
                group_result = self.evaluate_file_group(file_group, task_id, timestamp)
                group_result["point_name"] = point_name

                point_results["file_groups"].append(group_result)
                results["all_file_groups"].append(group_result)

                # Collect scores for point average
                group_weighted_score = group_result["group_best_weighted_score"]
                point_weighted_scores.append(group_weighted_score)

                # Update best results
                if group_weighted_score > best_weighted_score_in_point:
                    best_weighted_score_in_point = group_weighted_score
                    best_result_in_point = group_result["best_result"]

                # Update overall best
                if group_weighted_score > overall_best_weighted_score:
                    overall_best_weighted_score = group_weighted_score
                    overall_best_result = group_result["best_result"]

            # Calculate point average scores
            point_weighted_average = sum(point_weighted_scores) / len(point_weighted_scores) if point_weighted_scores else 0
            if len(point_weighted_scores) > 1:
                point_weighted_std = float(statistics.stdev(point_weighted_scores))
            else:
                point_weighted_std = 0.0

            point_results["best_result"] = best_result_in_point
            point_results["point_best_weighted_score"] = best_weighted_score_in_point
            point_results["point_average_weighted_score"] = point_weighted_average
            point_results["point_weighted_std"] = point_weighted_std  # Save std
            results["point_results"][point_name] = point_results

        # Calculate task-level statistics
        score_stats = self.calculate_task_average_score(results["all_file_groups"])

        # Calculate task-level std from all points
        all_point_stds = [
            point_data["point_weighted_std"]
            for point_data in results["point_results"].values()
            if "point_weighted_std" in point_data
        ]
        if len(all_point_stds) > 1:
            results["task_weighted_std"] = float(statistics.stdev(all_point_stds))
        elif len(all_point_stds) == 1:
            results["task_weighted_std"] = all_point_stds[0]
        else:
            results["task_weighted_std"] = 0.0

        results["overall_best"] = overall_best_result
        results["overall_best_weighted_score"] = overall_best_weighted_score
        results["task_average_weighted_score"] = score_stats["average_weighted_score"]
        results["score_statistics"] = score_stats
        results["total_file_groups_evaluated"] = len(results["all_file_groups"])

        return results
        


    def print_detailed_evaluation_summary(self, results: Dict) -> None:
        """
        Print a detailed summary of evaluation results with weighted scoring
        
        Args:
            results: Evaluation results dictionary
        """
        logging.info(f"\n📊 {results['task_id'].upper()} Detailed Evaluation Results")
        logging.info("=" * 80)
        
        logging.info(f"📁 Total File Groups Evaluated: {results.get('total_file_groups_evaluated', 0)}")
        
        # Print task-level statistics
        score_stats = results.get("score_statistics", {})
        logging.info(f"📈 Task Average Weighted Score: {results.get('task_average_weighted_score', 0):.1f}/100")
        logging.info(f"📊 Valid Groups (>0): {score_stats.get('valid_groups', 0)}/{score_stats.get('total_groups', 0)}")
        if "task_weighted_variance" in results:
            logging.info(f"📉 Task Weighted Variance: {results['task_weighted_variance']:.2f}")
        if score_stats.get('total_groups', 0) > 0:
            logging.info(f"📊 Weighted Score Range: {score_stats.get('min_weighted_score', 0):.1f} - {score_stats.get('max_weighted_score', 0):.1f}")
        if "task_weighted_std" in results:
            logging.info(f"📉 Task Weighted Std: {results['task_weighted_std']:.2f}")
       
        for point_name, point_data in results["point_results"].items():
            point_best_weighted_score = point_data.get("point_best_weighted_score", 0)
            point_avg_weighted_score = point_data.get("point_average_weighted_score", 0)
            point_std = point_data.get("point_weighted_std", 0.0)
            logging.info(f"\n📍 {point_name} (Best: {point_best_weighted_score:.1f}, Avg: {point_avg_weighted_score:.1f}, Std: {point_std:.2f}):")
        
        if results["overall_best"]:
            best = results["overall_best"]
            logging.info(f"\n🏆 OVERALL BEST RESULT:")
            logging.info(f"   🎯 Task ID: {best['task_id']}")
            logging.info(f"   🎪 Target Object: {best['target_name']}")
            logging.info(f"   🏅 Weighted Score: {best.get('weighted_score', 0):.1f}/100")
            # logging.info(f"   ⚠️ Recognition Error: {'Yes' if best.get('recognition_error', False) else 'No'}")
            logging.info(f"   📏 Min Distance: {best.get('minimum_distance', 'N/A'):.2f} units" if best.get('minimum_distance') else "   📏 Distance: N/A")
            logging.info(f"   🔍 Strategy: {best.get('strategy', 'unknown')}")
            
            # Show distances to all targets and score breakdown
            if "distances_to_all_targets" in best and "score_breakdown" in best:
                distances = best["distances_to_all_targets"]
                target_count = best.get("target_count", len(distances))
                score_breakdown = best["score_breakdown"]
                
                logging.info(f"   🎯 Target Count: {target_count}")
                logging.info(f"   📏 Distances to All Targets: {[f'{d:.2f}' for d in distances]}")
                
                if score_breakdown and "score_breakdown" in score_breakdown:
                    logging.info(f"   📊 Score Breakdown:")
                    for i, breakdown in enumerate(score_breakdown["score_breakdown"]):
                        logging.info(f"      Point {i+1}: Distance={breakdown['distance']:.2f}, "
                              f"Score={breakdown['individual_score']:.1f}, "
                              f"Weight={breakdown['weight_percentage']:.1f}%, "
                              f"Weighted={breakdown['weighted_score']:.1f}")
            
            logging.info(f"   📍 AUV Position: {best.get('closest_auv_location', 'N/A')}")
            logging.info(f"   📍 AUV Location Index: {best.get('auv_location_index', 'N/A')}")
            logging.info(f"   🎯 Target Coordinate: {best.get('closest_target_coordinate', 'N/A')}")
            logging.info(f"   📄 Source File: {best.get('file_name', 'N/A')}")
            logging.info(f"   📂 File Type: {best.get('file_type', 'unknown')}")
            logging.info(f"   📍 Point: {best.get('point_name', 'N/A')}")
            logging.info(f"   🕐 Timestamp: {best.get('timestamp', 'N/A')}")
            
            # Show target positions
            if "target_info" in best:
                target_info = best["target_info"]
                logging.info(f"   🎯 Target Positions:")
                for pos_info in target_info.get("target_positions", []):
                    logging.info(f"      Point {pos_info['point_index']}: {pos_info['position_string']}")
        else:
            logging.info("❌ No valid results found")
        
        logging.info(f"\n📋 FILE GROUPS BY TIMESTAMP:")
        logging.info("-" * 80)
        
        # Group by point for better organization
        for point_name, point_data in results["point_results"].items():
            point_best_weighted_score = point_data.get("point_best_weighted_score", 0)
            point_avg_weighted_score = point_data.get("point_average_weighted_score", 0)
            point_variance = point_data.get("point_weighted_variance", 0.0)
            logging.info(f"\n📍 {point_name} (Best: {point_best_weighted_score:.1f}, Avg: {point_avg_weighted_score:.1f}, Variance: {point_variance:.2f}):")
            logging.info(f"    📊 Total File Groups: {point_data.get('total_file_groups', 0)}")
            logging.info(f"    📊 Important Memory Files: {point_data.get('important_memory_files', 0)}")
            logging.info(f"    📊 Regular Memory Files: {point_data.get('memory_files', 0)}")
            
            for idx, group_result in enumerate(point_data["file_groups"], 1):
                timestamp = group_result["timestamp"]
                group_weighted_score = group_result.get("group_best_weighted_score", 0)
                has_error = group_result.get("has_recognition_error", False)
                error_indicator = " ⚠️" if has_error else ""
                
                logging.info(f"\n   🕐 Group {idx}: {timestamp} (Score: {group_weighted_score:.1f}){error_indicator}")
                
                # Show results for each file type in the group
                for file_type, file_result in group_result["file_results"].items():
                    strategy = file_result.get('strategy', 'unknown')
                    file_weighted_score = file_result.get('weighted_score', 0)
                    file_recognition_error = file_result.get('recognition_error', False)
                    
                    logging.info(f"      📄 {file_type.upper()}: {file_result.get('file_name', 'Unknown')}")
                    logging.info(f"         🔍 Strategy: {strategy}")
                    
                    if "error" in file_result:
                        logging.info(f"         ❌ Error: {file_result['error']}")
                        logging.info(f"         🏅 Score: 0/100")
                    else:
                        logging.info(f"         🏅 Weighted Score: {file_weighted_score:.1f}/100")
                        # if file_recognition_error:
                        #     logging.info(f"         ⚠️ Recognition Error: Score < 30 but success returned")
                        if file_result.get('minimum_distance') is not None:
                            logging.info(f"         📏 Min Distance: {file_result['minimum_distance']:.2f} units")
                        if "distances_to_all_targets" in file_result:
                            distances = file_result["distances_to_all_targets"]
                            logging.info(f"         📏 All Distances: {[f'{d:.2f}' for d in distances]}")
                        if "auv_location_index" in file_result:
                            logging.info(f"         📍 Used Location Index: {file_result['auv_location_index']}")
                        logging.info(f"         📊 Total Locations: {file_result.get('total_locations', 0)}")
                        logging.info(f"         📊 Analyzed Locations: {file_result.get('locations_analyzed', 0)}")
                        logging.info(f"         🎪 Target: {file_result.get('target_name', 'Unknown')}")
                
                # Show group best result
                if group_result["best_result"]:
                    best = group_result["best_result"]
                    logging.info(f"      🏅 GROUP BEST: {best['file_type']} file - Score: {group_weighted_score:.1f}")
                else:
                    logging.info(f"      ❌ No valid results in this group")
                logging.info("")

    
    def save_detailed_results(self, all_results: Dict, base_path: str) -> None:
        """
        Save detailed results
        
        Args:
            all_results: All evaluation results
            base_path: Base path for saving files
        """
        # Save comprehensive results
        output_file = os.path.join(base_path, "eval", "auv_weighted_evaluation.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logging.info(f"\n💾 Weighted evaluation results saved to: {output_file}")
        except Exception as e:
            logging.error(f"❌ Error saving weighted results: {e}")
        
        # Save score summary
        score_summary_file = os.path.join(base_path, "eval", "weighted_score_summary.json")
        try:
            score_summary = {}
            for task_id, task_results in all_results.items():
                best_result = task_results.get("overall_best")
                score_stats = task_results.get("score_statistics", {})
                
                score_summary[task_id] = {
                    "task_id": task_id,
                    "overall_best_weighted_score": task_results.get("overall_best_weighted_score", 0),
                    "task_average_weighted_score": task_results.get("task_average_weighted_score", 0),
                    "score_statistics": score_stats,
                    "overall_best_distance": best_result.get("minimum_distance") if best_result else None,
                    "target_count": best_result.get("target_info", {}).get("target_count") if best_result else 0,
                    "target_name": best_result.get("target_name") if best_result else None,
                    "target_positions": best_result.get("target_info", {}).get("target_positions", []) if best_result else [],
                    "total_file_groups": task_results.get("total_file_groups_evaluated", 0),
                    "recognition_errors": score_stats.get("recognition_errors", 0),
                    "best_file_info": {
                        "file_name": best_result.get("file_name"),
                        "file_type": best_result.get("file_type"),
                        "strategy": best_result.get("strategy"),
                        "point_name": best_result.get("point_name"),
                        "timestamp": best_result.get("timestamp"),
                        "recognition_error": best_result.get("recognition_error", False),
                        "auv_location_index": best_result.get("auv_location_index")
                    } if best_result else None
                }
            
            with open(score_summary_file, 'w', encoding='utf-8') as f:
                json.dump(score_summary, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 Weighted score summary saved to: {score_summary_file}")
        except Exception as e:
            logging.error(f"❌ Error saving weighted score summary: {e}")


def main():
    eval_root = os.path.join(base_path, "eval", "navigation")
    eval_roots = [
        os.path.join(eval_root, "main", "gpt4omini"),
        os.path.join(eval_root, "main", "gemini"),
        os.path.join(eval_root, "main", "qwen"),
        os.path.join(eval_root, "migration", "gpt4o"),
        os.path.join(eval_root, "migration", "qwen"),

        os.path.join(eval_root, "scale","qwen"),

    ]

    for eval_root in eval_roots:
        if not os.path.exists(eval_root):
            print(f"Folder does not exist: {eval_root}")
            continue

        logger, log_filepath = setup_logging(eval_root)
        logging.info(f"\n🚀 Starting AUV Performance Evaluation for: {eval_root}")

        task_dirs = glob.glob(os.path.join(eval_root, "task*"))
        evaluator = AUVPerformanceEvaluator()
        all_results = {}

        for task_dir in sorted(task_dirs):
            task_id = os.path.basename(task_dir)
            logging.info(f"\n🔍 Evaluating Task: {task_id}")
            task_results = evaluator.evaluate_task_directory(task_dir)
            evaluator.print_detailed_evaluation_summary(task_results)
            all_results[task_id] = task_results

        # save detailed results
        evaluator.save_detailed_results(all_results, eval_root)

        logging.info("\n✅ Weighted scoring evaluation completed!")
        logging.info(f"📝 Complete log saved to: {log_filepath}")

if __name__ == "__main__":
    main()