/************************************************************************************
***
*** Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Tue 02 Apr 2024 03:49:53 PM CST
***
************************************************************************************/

#include "facedet.h"
#include "facegan.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

#include <sys/stat.h> // for chmod

#include <eigen3/Eigen/Dense>

#define NMS_THRESHOLD 0.4
#define SCORE_THRESHOLD 0.8
#define MIN_EYES_THRESHILD 10 // 10 pixels
#define STANDARD_FACE_SIZE 512 // DO NOT Change it

typedef struct {
    float s, x, y, w, h; // s -- score
    int index;
} ScoreBox;

static float overlap(float x1, float w1, float x2, float w2);
static float box_iou(const ScoreBox& a, const ScoreBox& b);
static void nms_sort(std::vector<ScoreBox>& detboxes, float thresh);

static int make_anchor_boxes(std::vector<float>& anchor_boxes_vector, int h, int w, int s, int size1, int size2);
static int decode_detect_result(int h, int w, TENSOR* detect_result);
static int decode_landmarks(int h, int w, TENSOR* detect_result);

static TENSOR* standard_face_mask(int B, int C);
static Eigen::Matrix3f get_affine_matrix(float* landmarks);
static TENSOR* get_affine_grid(Eigen::Matrix3f theta, int OH, int OW);
static TENSOR* get_affine_image(TENSOR* input, Eigen::Matrix3f matrix, int OH, int OW);

static void output_noface_image(TENSOR* input_tensor, char* output_file);
static int to_bgr_image(TENSOR *x);
// -----------------------------------------------------------------------------------------

static float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_iou(const ScoreBox& a, const ScoreBox& b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0)
        return 0.0;
    float i = w * h;
    float u = a.w * a.h + b.w * b.h - i + 1e-5;

    return i / u;
}

static void nms_sort(std::vector<ScoreBox>& detboxes, float thresh)
{
    size_t total = detboxes.size();
    std::sort(detboxes.begin(), detboxes.begin() + total, [=](const ScoreBox& a, const ScoreBox& b) {
        return a.s > b.s;
    });
    for (size_t i = 0; i < total; ++i) {
        if (detboxes[i].s == 0)
            continue;

        ScoreBox a = detboxes[i];
        for (size_t j = i + 1; j < total; ++j) {
            ScoreBox b = detboxes[j];
            if (box_iou(a, b) > thresh)
                detboxes[j].s = 0;
        }
    }
}

static int make_anchor_boxes(std::vector<float>& anchor_boxes_vector, int H, int W, int S, int size1, int size2)
{
    int H2 = (int)(H + S - 1) / S;
    int W2 = (int)(W + S - 1) / S;
    float s_kx1 = float(size1) / W;
    float s_ky1 = float(size1) / H;
    float s_kx2 = float(size2) / W;
    float s_ky2 = float(size2) / H;

    for (int i = 0; i < H2; i++) {
        float cy = (i + 0.5) * (float)S / (float)H;
        for (int j = 0; j < W2; j++) {
            float cx = (j + 0.5) * (float)S / (float)W;
            anchor_boxes_vector.push_back(cx);
            anchor_boxes_vector.push_back(cy);
            anchor_boxes_vector.push_back(s_kx1);
            anchor_boxes_vector.push_back(s_ky1);
            // ----------------------------
            anchor_boxes_vector.push_back(cx);
            anchor_boxes_vector.push_back(cy);
            anchor_boxes_vector.push_back(s_kx2);
            anchor_boxes_vector.push_back(s_ky2);
        }
    }
    return RET_OK;
}

static Eigen::Matrix3f get_affine_matrix(float* landmarks)
{
    Eigen::MatrixXf Q(10, 4);
    Eigen::VectorXf S(10);

    // Q << 324.868744,  133.635284, 1.000000, 0.000000,
    //   133.635284, -324.868744, 0.000000, 1.000000,
    //   372.724915,  132.330826, 1.000000, 0.000000,
    //   132.330826, -372.724915, 0.000000, 1.000000,
    //   347.212708,  158.984650, 1.000000, 0.000000,
    //   158.984650, -347.212708, 0.000000, 1.000000,
    //   328.528015,  176.166473, 1.000000, 0.000000,
    //   176.166473, -328.528015, 0.000000, 1.000000,
    //   372.120667,  175.495361, 1.000000, 0.000000,
    //   175.495361, -372.120667, 0.000000, 1.000000;
    for (int i = 0; i < 5; i++) {
        Q(2 * i, 0) = landmarks[2 * i]; // x
        Q(2 * i, 1) = landmarks[2 * i + 1]; // y
        Q(2 * i, 2) = 1.0;
        Q(2 * i, 3) = 0.0;
        // ------------------------------------
        Q(2 * i + 1, 0) = landmarks[2 * i + 1]; // y
        Q(2 * i + 1, 1) = -landmarks[2 * i]; // -x
        Q(2 * i + 1, 2) = 0.0;
        Q(2 * i + 1, 3) = 1.0;
    }
    // Standard face (512x512) landmarks
    S << 192.981384, 239.947083,
        318.902771, 240.193604,
        256.634155, 314.019348,
        201.261169, 371.410431,
        313.089050, 371.151184;

    Eigen::Vector4f M = Q.colPivHouseholderQr().solve(S);

    Eigen::Matrix3f matrix(3, 3);
    matrix << M(0), M(1), M(2), -M(1), M(0), M(3), 0.0, 0.0, 1.0;
    // 2.80723, -0.0635898, -713.528, -150.88

    return matrix;
}

static TENSOR* get_affine_grid(Eigen::Matrix3f theta, int OH, int OW)
{
    TENSOR* grid = tensor_make_grid(1, OH, OW); // 1x2xOHxOW
    CHECK_TENSOR(grid);

    for (int i = 0; i < OH; i++) {
        float* row_y = tensor_start_row(grid, 0, 0, i);
        float* row_x = tensor_start_row(grid, 0, 1, i);
        for (int j = 0; j < OW; j++) {
            // (row_x[j], row_y[j], 1.0) * theta.t()
            float t_x = row_x[j] * theta(0, 0) + row_y[j] * theta(0, 1) + 1.0 * theta(0, 2);
            float t_y = row_x[j] * theta(1, 0) + row_y[j] * theta(1, 1) + 1.0 * theta(1, 2);

            row_x[j] = (t_x + 1.0) * 0.5; // convert value from [-1.0, 1.0] to [0.0, 1.0] for tensor_grid_sample
            row_y[j] = (t_y + 1.0) * 0.5;
        }
    }

    return grid;
}

static TENSOR* get_affine_image(TENSOR* input, Eigen::Matrix3f matrix, int OH, int OW)
{
    Eigen::Matrix3f T1;
    T1 << 2.0 / input->width, 0.0, -1.0,
        0.0, 2.0 / input->height, -1.0,
        0.0, 0.0, 1.0;
    Eigen::Matrix3f T2;
    T2 << 2.0 / OW, 0.0, -1.0,
        0.0, 2.0 / OH, -1.0,
        0.0, 0.0, 1.0;
    Eigen::Matrix3f T3 = T2 * matrix * T1.inverse();
    Eigen::Matrix3f theta = T3.inverse();
    // theta:
    //   0.364586 0.00825855   0.393891
    //  -0.0117643 0.519353  -0.219108
    //  0.0         0.0          1.0

    TENSOR* grid = get_affine_grid(theta, OH, OW); // 1x2xOHxOW
    CHECK_TENSOR(grid);
    TENSOR* output = tensor_grid_sample(input, grid);
    tensor_destroy(grid);

    return output;
}

static int decode_detect_result(int h, int w, TENSOR* detect_result)
{
    check_tensor(detect_result);

    std::vector<float> anchor_boxes_vector;
    make_anchor_boxes(anchor_boxes_vector, h, w, 8, 16, 32);
    make_anchor_boxes(anchor_boxes_vector, h, w, 16, 64, 128);
    make_anchor_boxes(anchor_boxes_vector, h, w, 32, 256, 512);

    TENSOR* anchor_boxes_tensor = tensor_create(1, 1, anchor_boxes_vector.size() / 4, 4);
    check_tensor(anchor_boxes_tensor);
    memcpy(anchor_boxes_tensor->data, anchor_boxes_vector.data(), anchor_boxes_vector.size() * sizeof(float));
    anchor_boxes_vector.clear();

    // Decode boxes
    float lin_scale = 0.1;
    float exp_scale = 0.2;
    float *boxes_row, *anchors_row, *landmarks_row;
    for (int i = 0; i < detect_result->height; i++) {
        float b[4];
        anchors_row = tensor_start_row(anchor_boxes_tensor, 0, 0, i);
        boxes_row = tensor_start_row(detect_result, 0, 0, i) + 2; // boxes offset == 2 in detect_result

        b[0] = anchors_row[0] + boxes_row[0] * lin_scale * anchors_row[2];
        b[1] = anchors_row[1] + boxes_row[1] * lin_scale * anchors_row[3];
        b[2] = anchors_row[2] * exp(boxes_row[2] * exp_scale);
        b[3] = anchors_row[3] * exp(boxes_row[3] * exp_scale);

        // boxes: (cx, cy, w, h) ==> (x1, y1, x2, y2)
        boxes_row[0] = b[0] - b[2] / 2.0;
        boxes_row[1] = b[1] - b[3] / 2.0;
        boxes_row[2] = b[0] + b[2] / 2.0;
        boxes_row[3] = b[1] + b[3] / 2.0;

        // Scale to real size
        boxes_row[0] *= w;
        boxes_row[1] *= h;
        boxes_row[2] *= w;
        boxes_row[3] *= h;
        for (int j = 0; j < 4; j++) {
            boxes_row[j] = MAX(boxes_row[j], 0.0);
            if (j % 2 == 0)
                boxes_row[j] = MIN(boxes_row[j], w);
            else
                boxes_row[j] = MIN(boxes_row[j], h);
        }
    }

    // Decode landmars
    for (int i = 0; i < detect_result->height; i++) {
        float t[10];
        anchors_row = tensor_start_row(anchor_boxes_tensor, 0, 0, i);
        landmarks_row = tensor_start_row(detect_result, 0, 0, i) + 6; // detect_result start == 6 in detect_result

        // t[0] = anchors_row[0] + landmarks_row[0] * lin_scale * anchors_row[2];
        // t[1] = anchors_row[1] + landmarks_row[1] * lin_scale * anchors_row[3];
        // t[2] = anchors_row[0] + landmarks_row[2] * lin_scale * anchors_row[2];
        // t[3] = anchors_row[1] + landmarks_row[3] * lin_scale * anchors_row[3];
        // t[4] = anchors_row[0] + landmarks_row[4] * lin_scale * anchors_row[2];
        // t[5] = anchors_row[1] + landmarks_row[5] * lin_scale * anchors_row[3];
        // t[6] = anchors_row[0] + landmarks_row[6] * lin_scale * anchors_row[2];
        // t[7] = anchors_row[1] + landmarks_row[7] * lin_scale * anchors_row[3];
        // t[8] = anchors_row[0] + landmarks_row[8] * lin_scale * anchors_row[2];
        // t[9] = anchors_row[1] + landmarks_row[9] * lin_scale * anchors_row[3];
        for (int j = 0; j < 10; j++) {
            if (j % 2 == 0) {
                t[j] = anchors_row[0] + landmarks_row[j] * lin_scale * anchors_row[2];
                landmarks_row[j] = MIN(MAX(t[j] * w, 0.0), w);
            } else {
                t[j] = anchors_row[1] + landmarks_row[j] * lin_scale * anchors_row[3];
                landmarks_row[j] = MIN(MAX(t[j] * h, 0.0), h);
            }
        }
    }

    tensor_destroy(anchor_boxes_tensor);

    return RET_OK;
}

static int decode_landmarks(int h, int w, TENSOR* detect_result)
{
    int n_detected_face = 0;
    decode_detect_result(h, w, detect_result);

    std::vector<ScoreBox> det_boxes;
    for (int i = 0; i < detect_result->height; i++) {
        float* row = tensor_start_row(detect_result, 0, 0, i);
        ScoreBox box;
        box.s = row[1]; // fg score
        box.x = row[2];
        box.y = row[3];
        box.w = row[4] - row[2];
        box.h = row[5] - row[3];
        box.index = i;
        if (box.s < SCORE_THRESHOLD || box.w < 0.5 || box.h < 0.5)
            continue;

        det_boxes.push_back(box);
    }
    nms_sort(det_boxes, NMS_THRESHOLD);

    // Update detect result score
    for (int i = 0; i < detect_result->height; i++) {
        float* row = tensor_start_row(detect_result, 0, 0, i);
        row[1] = 0.0; // clear all scores
    }
    for (auto it = det_boxes.begin(); it != det_boxes.end(); it++) {
        if (it->s >= SCORE_THRESHOLD) {
            n_detected_face++;
            float* row = tensor_start_row(detect_result, 0, 0, it->index);
            row[1] = it->s; // update score
        }
    }
    det_boxes.clear();

    return n_detected_face;
}

static TENSOR* standard_face_mask(int B, int C)
{
    static float pad[20] = {
        0.000000, 0.000000, 0.000000, 0.013998, 0.130770, 0.367597, 0.484369, 0.498367, 0.512365, 0.629137,
        0.865964, 0.982736, 0.996734, 0.996734, 0.996734, 0.996734, 0.996734, 0.996734, 0.996734, 0.996734
    };

    TENSOR* mask = tensor_create(B, C, STANDARD_FACE_SIZE, STANDARD_FACE_SIZE);
    CHECK_TENSOR(mask);

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < STANDARD_FACE_SIZE; i++) {
                float* row = tensor_start_row(mask, b, c, i);
                for (int j = 0; j < STANDARD_FACE_SIZE; j++) {
                    int d = MIN(i, j); // to border distance
                    d = MIN(d, STANDARD_FACE_SIZE - i);
                    d = MIN(d, STANDARD_FACE_SIZE - j);
                    row[j] = (d < ARRAY_SIZE(pad)) ? pad[d] : 1.0;
                }
            }
        }
    }

    return mask;
}

static void output_noface_image(TENSOR* input_tensor, char* output_file)
{
    IMAGE* image = image_from_tensor(input_tensor, 0 /*batch*/);
    check_avoid(image_valid(image));

    image_drawtext(image, 10, 10, (char*)"NO Face", 0xFF0000);
    image_save(image, output_file);
    image_destroy(image);
}

static int to_bgr_image(TENSOR *x)
{
    float *R, *G, *B, r, b;

    check_tensor(x);
    check_point(x->chan >= 3);

    R = x->data;
    G = R + x->height * x->width;
    B = G + x->height * x->width;

    // RGB mean - [0.4823, 0.4588, 0.4078]
    for (int h = 0; h < x->height; h++) {
        for (int w = 0; w < x->width; w++) {
            r = (*R - 0.4823) * 255.0;
            *G = (*G - 0.4588) * 255;
            b = (*B - 0.4078) * 255.0;
            *R = b; *B = r; // swap R/B for BGR
            R++; G++; B++;
        }
    }

    return RET_OK;
}

TENSOR *facedet_forward(RetinaFace *net, TENSOR *input_tensor)
{
    CHECK_TENSOR(input_tensor);

    TENSOR *bgr_tensor = tensor_copy(input_tensor);
    CHECK_TENSOR(bgr_tensor);
    to_bgr_image(bgr_tensor);
    TENSOR *argv[1];
    argv[0] = bgr_tensor ;
    TENSOR *output_tensor = net->engine_forward(ARRAY_SIZE(argv), argv);
    tensor_destroy(bgr_tensor);

    // TENSOR *xxxx_test = net->get_output_tensor("x0");
    // if (tensor_valid(xxxx_test)) {
    //     tensor_show("********************** x0", xxxx_test);
    //     tensor_destroy(xxxx_test);
    // }
	return output_tensor;
}

TENSOR *facegan_forward(CodeFormer *net, TENSOR *input_tensor)
{
    CHECK_TENSOR(input_tensor);

    TENSOR *argv[1];
    argv[0] = input_tensor ;

	TENSOR *output_tensor = net->engine_forward(ARRAY_SIZE(argv), argv);
    // TENSOR *xxxx_test = net->get_output_tensor("x0");
    // if (tensor_valid(xxxx_test)) {
    //     tensor_show("********************** x0", xxxx_test);
    //     tensor_destroy(xxxx_test);
    // }
	return output_tensor;
}

int face_detect(RetinaFace *net, char *input_file, char *output_file)
{
    TENSOR* input_tensor;
	printf("Face detect %s to %s ...\n", input_file, output_file);

    input_tensor = tensor_load_image(input_file, 0 /*with_alpha*/);
    check_tensor(input_tensor);

    TENSOR* detect_result = facedet_forward(net, input_tensor); // 1x1xnx16
    check_tensor(detect_result);

    int n = decode_landmarks(input_tensor->height, input_tensor->width, detect_result);
    if (n < 1) { // NOT Found face
        output_noface_image(input_tensor, output_file);
    } else { // Checking face ...
        int nface = 0;
        TENSOR* grid_tensors[16];

        for (int i = 0; i < detect_result->height && nface < ARRAY_SIZE(grid_tensors); i++) {
            float* row = tensor_start_row(detect_result, 0, 0, i);
            if (row[1] < SCORE_THRESHOLD)
                continue;

            float* landmarks = row + 6;
            float eye_dist = abs(landmarks[0] - landmarks[2]); // left eye (x1, y1), right eye (x2, y2)
            if (eye_dist < MIN_EYES_THRESHILD) // Skip strange face ...
                continue;
            Eigen::Matrix3f matrix = get_affine_matrix(landmarks);

            grid_tensors[nface] = get_affine_image(input_tensor, matrix, STANDARD_FACE_SIZE, STANDARD_FACE_SIZE);
            check_tensor(grid_tensors[nface]);
            nface++;
        }
        if (nface < 1) {
            output_noface_image(input_tensor, output_file);
        } else {
            tensor_saveas_grid(nface, grid_tensors, output_file);
            for (int i = 0; i < nface; i++) {
                tensor_destroy(grid_tensors[i]);
            }
        }
    }
    chmod(output_file, 0644);

    tensor_destroy(detect_result);
    tensor_destroy(input_tensor);

    return RET_OK;
}

int face_gan(CodeFormer *net, TENSOR* input, TENSOR* detect_result)
{
    check_tensor(input);
    check_tensor(detect_result);
    tensor_show("detect_result", detect_result);

    TENSOR* face_mask = standard_face_mask(input->batch, input->chan);
    check_tensor(face_mask);

    for (int i = 0; i < detect_result->height; i++) {
        float* row = tensor_start_row(detect_result, 0, 0, i);
        if (row[1] < SCORE_THRESHOLD)
            continue;

        float* landmarks = row + 6;
        float eye_dist = abs(landmarks[0] - landmarks[2]); // left eye (x1, y1), right eye (x2, y2)
        if (eye_dist < MIN_EYES_THRESHILD) // Skip strange face ...
            continue;
        Eigen::Matrix3f matrix = get_affine_matrix(landmarks);

        TENSOR* cropped_face = get_affine_image(input, matrix, STANDARD_FACE_SIZE, STANDARD_FACE_SIZE);
        check_tensor(cropped_face);

        TENSOR* refined_face = facegan_forward(net, cropped_face);
        check_tensor(refined_face);

        // RM = torch.linalg.inv(M)
        // pasted_face = get_affine_image(refined_face, RM, H, W)
        // pasted_mask = get_affine_image(self.face_mask, RM, H, W)
        // x = (1.0 - pasted_mask) * x + pasted_mask * pasted_face
        Eigen::Matrix3f matrix_inverse = matrix.inverse();
        TENSOR* pasted_face = get_affine_image(refined_face, matrix_inverse, input->height, input->width);
        check_tensor(pasted_face);
        TENSOR* pasted_mask = get_affine_image(face_mask, matrix_inverse, input->height, input->width);
        check_tensor(pasted_mask);
        for (int i = 0; i < input->batch * input->chan * input->height * input->width; i++) {
            input->data[i] = (1.0 - pasted_mask->data[i]) * input->data[i] + pasted_mask->data[i] * pasted_face->data[i];
        }
        tensor_destroy(pasted_mask);
        tensor_destroy(pasted_face);

        tensor_destroy(refined_face);
        tensor_destroy(cropped_face);
    }
    tensor_destroy(face_mask);

    return RET_OK;
}

int face_beauty(RetinaFace *det_net, CodeFormer *gan_net, char *input_file, char *output_file)
{
    TENSOR* input_tensor;

    printf("Face beauty %s to %s ...\n", input_file, output_file);

    input_tensor = tensor_load_image(input_file, 0 /*with_alpha*/);
    check_tensor(input_tensor);

    TENSOR* detect_result = facedet_forward(det_net, input_tensor); // 1x1xnx16
    check_tensor(detect_result);

    int n = decode_landmarks(input_tensor->height, input_tensor->width, detect_result);
    CheckPoint("detect %d faces", n);

    if (n < 1) { // NOT Found face
        syslog_info("NOT Found face in '%s' ...", input_file);
        tensor_saveas_image(input_tensor, 0 /*batch*/, output_file);
    } else {
        face_gan(gan_net, input_tensor, detect_result);
        tensor_saveas_image(input_tensor, 0 /*batch*/, output_file);
    }
    chmod(output_file, 0644);
    // time_spend((char*)key);

    tensor_destroy(detect_result);
    tensor_destroy(input_tensor);

    return RET_OK;
}


int image_face_detect(int device, int argc, char** argv, char *output_dir)
{
    RetinaFace net;
    char *p, output_filename[1024];

    // load net weight ...
    GGMLModel model;
    {
        check_point(model.preload("models/image_facedet_f32.gguf") == RET_OK);
        net.set_device(device);
        net.start_engine();
    }

    for (int i = 0; i < argc; i++) {
        p = strrchr(argv[i], '/');
        if (p != NULL) {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, p + 1);
        } else {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, argv[i]);
        }

        net.load_weight(&model, "");

        face_detect(&net, argv[i], output_filename);
    }

    // free network ...
    {
        model.clear();
        net.stop_engine();
    }

    return RET_OK;
}

int image_face_beauty(int device, int argc, char** argv, char *output_dir)
{
    RetinaFace det_net;
    CodeFormer gan_net;

    char *p, output_filename[1024];

    // load net weight ...
    GGMLModel det_model;
    GGMLModel gan_model;
    {
        check_point(det_model.preload("models/image_facedet_f32.gguf") == RET_OK);
        check_point(gan_model.preload("models/image_facegan_f32.gguf") == RET_OK);

        // -----------------------------------------------------------------------------------------
        det_net.set_device(device);
        det_net.start_engine();

        gan_net.set_device(device);
        gan_net.start_engine();
    }

    for (int i = 0; i < argc; i++) {
        p = strrchr(argv[i], '/');
        if (p != NULL) {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, p + 1);
        } else {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, argv[i]);
        }

        det_net.load_weight(&det_model, "");
        gan_net.load_weight(&gan_model, "");

		face_beauty(&det_net, &gan_net, argv[i], output_filename);
    }

    // free network ...
    {
        gan_model.clear();
        det_model.clear();

        gan_net.stop_engine();
        det_net.stop_engine();
    }

    return RET_OK;
}
