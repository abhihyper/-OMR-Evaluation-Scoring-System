import cv2
import numpy as np
import argparse
import imutils
from imutils.perspective import four_point_transform

class OMRGrader:
    def __init__(self, answer_key, total_questions=50, options_per_question=4):
        self.answer_key = answer_key
        self.total_questions = total_questions
        self.options_per_question = options_per_question
    
    def preprocess_image(self, image_path):
        # Load image and convert to grayscale
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
            
        # Resize image to a standard width for consistent processing
        image = imutils.resize(image, width=700)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edged = cv2.Canny(blurred, 75, 200)
        
        return image, gray, edged
    
    def find_answer_sheet_contour(self, edged):
        # Find contours in the edge map
        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Initialize the contour corresponding to the answer sheet
        doc_contour = None
        
        # Ensure at least one contour was found
        if len(contours) > 0:
            # Sort the contours by area in descending order
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Loop over the sorted contours
            for contour in contours:
                # Approximate the contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # If the approximated contour has four points, assume it's the answer sheet
                if len(approx) == 4:
                    doc_contour = approx
                    break
        
        return doc_contour
    
    def extract_answer_sheet(self, image, doc_contour):
        # Apply a four-point perspective transform to get a top-down view of the answer sheet
        paper = four_point_transform(image, doc_contour.reshape(4, 2))
        
        # Convert to grayscale and threshold
        warped = four_point_transform(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), doc_contour.reshape(4, 2))
        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        return paper, thresh
    
    def find_bubbles(self, thresh):
        # Find contours in the thresholded image
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        question_contours = []
        
        # Loop over the contours
        for contour in contours:
            # Compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Verify the contour is a bubble (circular/oval shape)
            if w >= 20 and h >= 20 and 0.9 <= aspect_ratio <= 1.1:
                question_contours.append(contour)
        
        # Sort the question contours from top to bottom
        question_contours = self.sort_contours(question_contours, method="top-to-bottom")
        
        return question_contours
    
    def sort_contours(self, contours, method="left-to-right"):
        # Initialize the reverse flag and sort index
        reverse = False
        i = 0
        
        # Handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
            
        # Handle if we are sorting against the y-coordinate rather than the x-coordinate
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
            
        # Construct the list of bounding boxes and sort them from top to bottom
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
            key=lambda b:b[1][i], reverse=reverse))
            
        return contours
    
    def grade_answer_sheet(self, image_path):
        # Preprocess the image
        image, gray, edged = self.preprocess_image(image_path)
        
        # Find the answer sheet contour
        doc_contour = self.find_answer_sheet_contour(edged)
        
        if doc_contour is None:
            print("Could not find answer sheet contour")
            return None, None
        
        # Extract the answer sheet
        paper, thresh = self.extract_answer_sheet(image, doc_contour)
        
        # Find bubbles
        question_contours = self.find_bubbles(thresh)
        
        # Check if we found the expected number of bubbles
        expected_bubbles = self.total_questions * self.options_per_question
        if len(question_contours) != expected_bubbles:
            print(f"Warning: Expected {expected_bubbles} bubbles, found {len(question_contours)}")
        
        # Group the question contours into rows (questions)
        questions = []
        bubbles_per_row = self.options_per_question
        
        for i in range(0, len(question_contours), bubbles_per_row):
            row = question_contours[i:i + bubbles_per_row]
            questions.append(row)
        
        # Initialize variables for scoring
        correct = 0
        answers = []
        
        # Loop over each question
        for q, question in enumerate(questions):
            # Sort the contours for this question from left to right
            question = self.sort_contours(question, method="left-to-right")
            
            # Initialize variables for bubble detection
            bubbled = None
            max_pixels = 0
            
            # Loop over each bubble in the question
            for j, bubble in enumerate(question):
                # Construct a mask for the bubble
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [bubble], -1, 255, -1)
                
                # Apply the mask to the thresholded image
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                
                # Count the number of non-zero pixels in the bubble area
                total = cv2.countNonZero(mask)
                
                # If this bubble has the most non-zero pixels, mark it as the selected answer
                if total > max_pixels:
                    max_pixels = total
                    bubbled = j
            
            # Determine if the answer is correct
            if bubbled is not None and bubbled == self.answer_key[q]:
                correct += 1
                answers.append((q, bubbled, True))
            else:
                answers.append((q, bubbled, False))
        
        # Calculate the score
        score = (correct / self.total_questions) * 100
        
        return score, answers
    
    def visualize_results(self, image_path, answers, output_path=None):
        # Load the original image
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=700)
        
        # Preprocess to get the answer sheet
        _, _, edged = self.preprocess_image(image_path)
        doc_contour = self.find_answer_sheet_contour(edged)
        
        if doc_contour is None:
            print("Could not find answer sheet for visualization")
            return
        
        paper, _ = self.extract_answer_sheet(image, doc_contour)
        
        # Draw results on the paper
        for q, bubbled, is_correct in answers:
            # Get the bubble contours for this question
            question_contours = self.find_bubbles(cv2.threshold(
                cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY), 0, 255, 
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1])
            
            # Group into questions
            questions = []
            bubbles_per_row = self.options_per_question
            
            for i in range(0, len(question_contours), bubbles_per_row):
                row = question_contours[i:i + bubbles_per_row]
                questions.append(row)
            
            if q < len(questions):
                question = questions[q]
                question = self.sort_contours(question, method="left-to-right")
                
                # Draw the correct/incorrect indication
                color = (0, 255, 0) if is_correct else (0, 0, 255)  # Green for correct, red for incorrect
                
                if bubbled is not None and bubbled < len(question):
                    # Draw contour around the selected answer
                    cv2.drawContours(paper, [question[bubbled]], -1, color, 3)
        
        # Display or save the result
        if output_path:
            cv2.imwrite(output_path, paper)
        
        cv2.imshow("Graded Paper", paper)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Example usage
    # Define the answer key (0-indexed, where 0=A, 1=B, 2=C, 3=D)
    answer_key = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                  0, 1, 2, 3, 0, 1, 2, 3, 0, 1]  # 50 questions
    
    # Initialize the OMR grader
    omr = OMRGrader(answer_key, total_questions=50, options_per_question=4)
    
    # Grade the answer sheet
    score, answers = omr.grade_answer_sheet("answer_sheet.jpg")
    
    if score is not None:
        print(f"Score: {score:.2f}%")
        
        # Display detailed results
        for q, bubbled, is_correct in answers:
            status = "CORRECT" if is_correct else "INCORRECT"
            selected = chr(65 + bubbled) if bubbled is not None else "NONE"
            correct = chr(65 + answer_key[q])
            print(f"Q{q+1}: Selected {selected}, Correct {correct} - {status}")
        
        # Visualize the results
        omr.visualize_results("answer_sheet.jpg", answers, "graded_sheet.jpg")
    else:
        print("Grading failed")

if __name__ == "__main__":
    main()