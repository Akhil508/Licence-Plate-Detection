import cv2
import pytesseract
import numpy

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'


def canny(image):
    return cv2.Canny(image, 150, 250)

if __name__ == "__main__":

    cap = cv2.VideoCapture('VID_20200706_154019.mp4')

    while cv2.waitKey(1) < 0:
        hasFrame, image = cap.read()
        orig = image
        (H, W) = image.shape[:2]

        (newW, newH) = (640, 320)
        rW = W / float(newW)
        rH = H / float(newH)

        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        #convert to gray
        Gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Gray",Gray)

        #generating blur and canny image
        blur = cv2.bilateralFilter(Gray,11,10,10)
        edges = canny(blur)
        #cv2.imshow('Result gray',edges)

        #cv2.imshow('blur',blur)
        cont,new = cv2.findContours(edges.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #print(cont[3])

        image_copy = edges.copy()
        _ = cv2.drawContours(image_copy,cont,-1,(255,0,255),2)
        #cv2.imshow('contours',image_copy)
        #print(len(cont))

        cont = sorted(cont, key=cv2.contourArea ,reverse=True)[:35]
        image_sorted = edges.copy()
        _ = cv2.drawContours(image_sorted,cont,-1,(255,0,255),2)
        cv2.imshow('sort',image_sorted)

        #ret,thresh = cv2.threshold(Gray,200,255,cv2.THRESH_BINARY)
        licenceplate = None
        for c in cont:
            perimeter = cv2.arcLength(c,True)
            edge_count = cv2.approxPolyDP(c,0.02*perimeter,True)
            if len(edge_count)==4:
                #contour = edge_count
                x,y,w,h =cv2.boundingRect(c)
                licenceplate = image[y:y+h,x:x+w]

                #licenceplate = cv2.cvtColor(text,cv2.COLOR_BGR2GRAY)
                #custom_config = r'--oem 3 --psm 6'
                textrecognized = pytesseract.image_to_string(licenceplate)

                # draw the bounding box on the image
                #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                orig = cv2.putText(image, textrecognized, (x +5,y + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2, cv2.LINE_AA)

                # draw the bounding box on the image
                #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
                orig = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.imshow("Text Detection", orig)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
    cv2.destroyAllWindows()
    cap.release()



