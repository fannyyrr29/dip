<?php
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    if (isset($_POST['base64Image'])) {
        $base64Image = $_POST['base64Image'];
        $imageData = base64_decode($base64Image);

        // Create a temporary file
        $tempImage = tmpfile();
        $metaDatas = stream_get_meta_data($tempImage);
        $tempFileName = $metaDatas['uri'];
        file_put_contents($tempFileName, $imageData);

        $cfile = curl_file_create($tempFileName, 'image/jpeg', 'upload.jpg');
        $data = array('file' => $cfile);

        // Flask API
        $ch = curl_init('http://127.0.0.1:5000/predict'); // Use your Flask API URL
        curl_setopt($ch, CURLOPT_POST, 1);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

        $response = curl_exec($ch);
        curl_close($ch);
        fclose($tempImage);

        $result = json_decode($response, true);

        if ($result) {
            echo json_encode($result);
        } else {
            echo json_encode(array('error' => 'Prediction failed.'));
        }
    } else {
        echo json_encode(array('error' => 'No image data found.'));
    }
} else {
    echo json_encode(array('error' => 'Invalid request.'));
}
