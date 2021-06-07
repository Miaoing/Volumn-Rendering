#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#define GL_SILENCE_DEPRECATION
#include <GL/glew.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glui.h>
#include <string.h>

#include <cstring>
#include <cmath>
#define PI 3.14159265359
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#define GL_ERROR() checkForOpenGLError(__FILE__, __LINE__)

using namespace std;
string name = "/Users/zhoutaotao/Downloads/Volumeric_Datasets_for_Topic_8/engine";
string fileName = name + ".raw";
using glm::mat4;
using glm::vec3;
GLuint g_vao;
GLuint g_programHandle;
GLuint g_winWidth = 400;
GLuint g_winHeight = 400;
GLdouble g_angle = 0;
GLuint g_frameBuffer;
// transfer function
GLuint g_tffTexObj;
GLuint g_bfTexObj;
GLuint g_texWidth;
GLuint g_texHeight;
GLuint g_volTexObj;
GLuint g_rcVertHandle;
GLuint g_rcFragHandle;
GLuint g_bfVertHandle;
GLuint g_bfFragHandle;

float g_stepSize = 0.001f;

int lastX, lastY;                // last mouse motion position
bool leftDown, middleDown, middleUp, shiftDown,scaleUp,scaleDown;        // mouse down and shift down flags
double sphi = 90.0, stheta = 45.0, sdepth = 0;    // for simple trackball
double xpan = 0.0, ypan = 0.0;                // for simple trackball
double zNear = 1.0, zFar = 100.0;
double scale = 1.0;

static GLdouble viewer[]= {-0.5, -0.5, -0.5}; /* initial viewer location */

// transfer function GLUI
int red_enabled =1;
int green_enabled =1;
int blue_enabled=1;
int opacity_scalar=1;
int redMean=200;
int greenMean=100;
int blueMean=50;
int opacityMean=128;
int redChannelWidth=30;
int greenChannelWidth=30;
int blueChannelWidth=30;
int opacityWidth=127;
int main_window;
int redStart, redEnd, greenStart, greenEnd, blueStart, blueEnd;
GLUI_Panel *red, *green, *blue, *opacity;
GLUI_Spinner *redSpinner, *greenSpinner, *blueSpinner;
GLUI_Scrollbar *redScrollbar, *greenScrollbar, *blueScrollbar;
#define RED_ENABLED_ID     100
#define GREEN_ENABLED_ID   101
#define BLUE_ENABLED_ID    102
#define RED_WIDTH_ID       103
#define GREEN_WIDTH_ID     104
#define BLUE_WIDTH_ID      105
// for opencv gui
using namespace cv;

int bg_w = 512; int bg_h = 400;
float left_pad = 0.15; float bottom_pad = 0.15;
int hist_w = bg_w * (1-2*left_pad); int hist_h = bg_h * (1-2*bottom_pad);
float origin[] = {bg_w * left_pad, (float)bg_h*(1-bottom_pad)}; // origin point

bool uniform = true, accumulate = false;
int cvblue[] = {255, 178, 102};
int cvgreen[] = {102, 255, 178};
int cvred[] = {153, 153, 255};
// init array
int histSize = 256;
float range[] = { 0, 256 }; //the upper boundary is exclusive
const float* histRange = { range };
const int num_bin = 256;
float bin[num_bin];
double bin_w = (double) hist_w / histSize;

void drawline(Mat img,Point start, Point end, int* color) {
    int thickness = 2;
    int linetype = 8;
    line(img, start, end, Scalar(color[0], color[1], color[2]), thickness, linetype);
 }

// guass distribution
float G(float x, float sigma, float mu) {
    float down = (1/(sigma*sqrt(2*PI)));
    float up = exp(-(x-mu)*(x-mu)/(2*sigma*sigma));
    float res = up / down;
    return res;
}

void drawcurve(Mat img, int* color, float sigma, float mu) {
    mu = mu / 256 * hist_w + origin[0];
    for (int i = 0; i < 256; i++) {
        Point s(origin[0] + bin_w*i, origin[1] - G(origin[0] + bin_w*i, sigma, mu));
        Point e(origin[0] + bin_w*(i+1), origin[1] - G(origin[0] + bin_w*(i+1), sigma, mu));
        drawline(img, s, e, color);
    }
}

void drawcoord(Mat img){
    // draw the origin
    circle(img, Point(origin[0], origin[1]), 3, Scalar(204,0,102), FILLED, 8, 0);
    Point s(origin[0], origin[1]);
    Point e(origin[0] + hist_w + 10, origin[1]); // x-axis
    arrowedLine(img, s, e, Scalar(0,0,0), 2, 8, 0, 0.04);

    Point s2(origin[0], origin[1]);
    Point e2(origin[0], origin[1] - hist_h -10);
    arrowedLine(img, s2, e2, Scalar(0,0,0), 2, 8, 0, 0.04);
    putText(img, //target image
                "Intensity", //text
                cv::Point(origin[0] + hist_w/4*3, origin[1]+20), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                0.7,
                CV_RGB(118, 185, 0), //font color
                2);
    putText(img, //target image
            "# of pixels", //text
            cv::Point(origin[0]-50, origin[1] - hist_h -20), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            0.7,
            CV_RGB(118, 185, 0), //font color
            2);

}

void hist(Mat histImage){
    ifstream myData(fileName, ios::binary);

    memset(bin, 0, sizeof(unsigned int)*num_bin);

    // read raw data
    int i = 0;
    uint8_t value;
    while (myData.read((char*)&value, 1))
    {
        bin[value] += 1;
        i++;
    }

    // set the range of the data
    for (int i = 0; i < 1; i++) {
        bin[i] = 0;
    }

    // normalize to 400
    int max = 0; int id = 0;
    for(int i = 0; i < 256; i++) {
        if(max < bin[i]) {
            max = bin[i];
            id = i;
        };
    }
    for(int i = 0; i < 256; i++) {
        bin[i] = bin[i] / max * hist_h;
    }
    for(int i = 0; i < num_bin; i++) {
        line(histImage, Point( origin[0]+i*bin_w, origin[1] - bin[i]),
                    Point( origin[0]+i*bin_w, origin[1]),
                    Scalar( 224, 224, 224), 4, 8, 0  );
    }
    drawcoord(histImage);
}



void MouseFunc(int button, int state, int x, int y) {

    lastX = x;
    lastY = y;
    leftDown = (button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN);
    middleDown = (button == GLUT_MIDDLE_BUTTON) && (state == GLUT_DOWN);
    middleUp = (button == GLUT_MIDDLE_BUTTON) && (state == GLUT_UP);
    shiftDown = (glutGetModifiers() & GLUT_ACTIVE_SHIFT);
    if (button == 3)    sdepth -= 0.05f;
    if (button == 4)    sdepth += 0.05f;
    glutPostRedisplay();
}


// GLUT mouse motion callback function
void MotionFunc(int x, int y) {
    if (leftDown)
        if (!shiftDown) { // rotate
            sphi += (double)(x - lastX) / 4.0;
            stheta += (double)(lastY - y) / 4.0;
        }
        else { // pan
            xpan += (double)(x - lastX)*sdepth / zNear / g_winWidth;
            ypan += (double)(lastY - y)*sdepth / zNear / g_winHeight;
        }
    lastX = x;
    lastY = y;
    glutPostRedisplay();
}

int checkForOpenGLError(const char* file, int line)
{
    // return 1 if an OpenGL error occured, 0 otherwise.
    GLenum glErr;
    int retCode = 0;

    glErr = glGetError();
    while(glErr != GL_NO_ERROR)
    {
    cout << "glError in file " << file
         << "@line " << line << gluErrorString(glErr) << endl;
    retCode = 1;
    exit(EXIT_FAILURE);
    }
    return retCode;
}
void keyboard(unsigned char key, int x, int y);
void display(void);
void initVBO();
void initShader();
void initFrameBuffer(GLuint, GLuint, GLuint);
GLuint initTFF1DTex();
GLuint initFace2DTex(GLuint texWidth, GLuint texHeight);
GLuint initVol3DTex(const char* filename, GLuint width, GLuint height, GLuint depth);
void render(GLenum cullFace);
void init(string fileName, int x, int y, int z)
{
    g_texWidth = g_winWidth;
    g_texHeight = g_winHeight;
    initVBO();
    initShader();
    g_tffTexObj = initTFF1DTex();
    g_bfTexObj = initFace2DTex(g_texWidth, g_texHeight);
    g_volTexObj = initVol3DTex((fileName).data(), x, y, z);
    initFrameBuffer(g_bfTexObj, g_texWidth, g_texHeight);
}
// init the vertex buffer object
void initVBO()
{
    GLfloat vertices[24] = {
    0.0, 0.0, 0.0,
    0.0, 0.0, 1.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 1.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 1.0,
    1.0, 1.0, 0.0,
    1.0, 1.0, 1.0
    };
// draw the six faces of the boundbox by drawwing triangles
// draw it contra-clockwise
// front: 1 5 7 3
// back: 0 2 6 4
// left: 0 1 3 2
// right:7 5 4 6
// up: 2 3 7 6
// down: 1 0 4 5
    GLuint indices[36] = {
    1,5,7,
    7,3,1,
    0,2,6,
    6,4,0,
    0,1,3,
    3,2,0,
    7,5,4,
    4,6,7,
    2,3,7,
    7,6,2,
    1,0,4,
    4,5,1
    };
    GLuint gbo[2];
    
    glGenBuffers(2, gbo);
    GLuint vertexdat = gbo[0];
    GLuint veridxdat = gbo[1];
    glBindBuffer(GL_ARRAY_BUFFER, vertexdat);
    glBufferData(GL_ARRAY_BUFFER, 24*sizeof(GLfloat), vertices, GL_STATIC_DRAW);
    // used in glDrawElement()
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, veridxdat);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 36*sizeof(GLuint), indices, GL_STATIC_DRAW);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    // vao like a closure binding 3 buffer object: verlocdat vercoldat and veridxdat
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0); // for vertexloc
    glEnableVertexAttribArray(1); // for vertexcol

    // the vertex location is the same as the vertex color
    glBindBuffer(GL_ARRAY_BUFFER, vertexdat);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLfloat *)NULL);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (GLfloat *)NULL);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, veridxdat);
    g_vao = vao;
}
void drawBox(GLenum glFaces)
{
    glEnable(GL_CULL_FACE);
    glCullFace(glFaces);
    glBindVertexArray(g_vao);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (GLuint *)NULL);
    glDisable(GL_CULL_FACE);
}
// check the compilation result
GLboolean compileCheck(GLuint shader)
{
    GLint err;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &err);
    if (GL_FALSE == err)
    {
    GLint logLen;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0)
    {
        char* log = (char *)malloc(logLen);
        GLsizei written;
        glGetShaderInfoLog(shader, logLen, &written, log);
        cerr << "Shader log: " << log << endl;
        free(log);
    }
    }
    return err;
}
// init shader object
GLuint initShaderObj(const GLchar* srcfile, GLenum shaderType)
{
    ifstream inFile(srcfile, ifstream::in);
    // use assert?
    if (!inFile)
    {
    cerr << "Error openning file: " << srcfile << endl;
    exit(EXIT_FAILURE);
    }
    
    const int MAX_CNT = 10000;
    GLchar *shaderCode = (GLchar *) calloc(MAX_CNT, sizeof(GLchar));
    inFile.read(shaderCode, MAX_CNT);
    if (inFile.eof())
    {
    size_t bytecnt = inFile.gcount();
    *(shaderCode + bytecnt) = '\0';
    }
    else if(inFile.fail())
    {
    cout << srcfile << "read failed " << endl;
    }
    else
    {
    cout << srcfile << "is too large" << endl;
    }
    // create the shader Object
    GLuint shader = glCreateShader(shaderType);
    if (0 == shader)
    {
    cerr << "Error creating vertex shader." << endl;
    }

    const GLchar* codeArray[] = {shaderCode};
    glShaderSource(shader, 1, codeArray, NULL);
    free(shaderCode);

    // compile the shader
    glCompileShader(shader);
    if (GL_FALSE == compileCheck(shader))
    {
    cerr << "shader compilation failed" << endl;
    }
    return shader;
}
GLint checkShaderLinkStatus(GLuint pgmHandle)
{
    GLint status;
    glGetProgramiv(pgmHandle, GL_LINK_STATUS, &status);
    if (GL_FALSE == status)
    {
    GLint logLen;
    glGetProgramiv(pgmHandle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0)
    {
        GLchar * log = (GLchar *)malloc(logLen);
        GLsizei written;
        glGetProgramInfoLog(pgmHandle, logLen, &written, log);
        cerr << "Program log: " << log << endl;
    }
    }
    return status;
}
// link shader program
GLuint createShaderPgm()
{
    // Create the shader program
    GLuint programHandle = glCreateProgram();
    if (0 == programHandle)
    {
    cerr << "Error create shader program" << endl;
    exit(EXIT_FAILURE);
    }
    return programHandle;
}


GLuint initTFF1DTex()
{
    GLuint tff1DTex;
    glGenTextures(1, &tff1DTex);
    glBindTexture(GL_TEXTURE_1D, tff1DTex);

    const int MAX_CNT = 10000;
    GLubyte *tff = (GLubyte *)calloc(MAX_CNT, sizeof(GLubyte));

    int opacityStart = (opacityMean > opacityWidth? opacityMean-opacityWidth: 0);
    int opacityEnd = (opacityMean + opacityWidth > 256? 256 : opacityMean+opacityWidth);
    for(int i=0; i<256; i++){
        tff[4*i]  = ((i>=redStart && i<= redEnd) ? (uint8_t)255 : (uint8_t)0);
        tff[4*i+1]= ((i>=greenStart && i<= greenEnd) ? (uint8_t)255 : (uint8_t)0);
        tff[4*i+2]= ((i>=blueStart && i<= blueEnd) ? (uint8_t)255 : (uint8_t)0);
        tff[4*i+3]= ((i>=opacityStart && i<= opacityEnd) ? (uint8_t)opacity_scalar : (uint8_t)0);
    }
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
 
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA16F, 12, 0, GL_RGBA, GL_UNSIGNED_BYTE, tff);
    
    free(tff);

    return tff1DTex;
}

// init the 2D texture for render backface 'bf' stands for backface
GLuint initFace2DTex(GLuint bfTexWidth, GLuint bfTexHeight)
{
    GLuint backFace2DTex;
    glGenTextures(1, &backFace2DTex);
    glBindTexture(GL_TEXTURE_2D, backFace2DTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, bfTexWidth, bfTexHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    return backFace2DTex;
}

// init 3D texture to store the volume data used fo ray casting
GLuint initVol3DTex(const char* filename, GLuint w, GLuint h, GLuint d)
{
    
    FILE *fp;
    size_t size = w * h * d;
    GLubyte *data = new GLubyte[size];              // 8bit
    if (!(fp = fopen(filename, "rb")))
    {
        cout << "Error: opening .raw file failed" << endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        cout << "OK: open .raw file successed" << endl;
    }
    if ( fread(data, sizeof(char), size, fp)!= size)
    {
        cout << "Error: read .raw file failed" << endl;
        exit(1);
    }
    else
    {
        cout << "OK: read .raw file successed" << endl;
    }
    fclose(fp);

    glGenTextures(1, &g_volTexObj);

    // bind 3D texture target
    glBindTexture(GL_TEXTURE_3D, g_volTexObj);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    // pixel transfer happens here from client to OpenGL server
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    GL_ERROR();
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, w, h, d, 0, GL_RED, GL_UNSIGNED_BYTE,data);
    GL_ERROR();
    delete []data;
    cout << "volume texture created" << endl;
    return g_volTexObj;
}

void checkFramebufferStatus()
{
    GLenum complete = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (complete != GL_FRAMEBUFFER_COMPLETE)
    {
    cout << "framebuffer is not complete" << endl;
    exit(EXIT_FAILURE);
    }
}
// init the framebuffer, the only framebuffer used in this program
void initFrameBuffer(GLuint texObj, GLuint texWidth, GLuint texHeight)
{
    // create a depth buffer for our framebuffer
    GLuint depthBuffer;
    glGenRenderbuffers(1, &depthBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, texWidth, texHeight);

    // attach the texture and the depth buffer to the framebuffer
    glGenFramebuffers(1, &g_frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, g_frameBuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texObj, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);
    checkFramebufferStatus();
    glEnable(GL_DEPTH_TEST);
}

void rcSetUinforms()
{
    // setting uniforms such as
    // ScreenSize
    // StepSize
    // TransferFunc
    // ExitPoints i.e. the backface, the backface hold the ExitPoints of ray casting
    // VolumeTex the texture that hold the volume data i.e. head256.raw
    GLint screenSizeLoc = glGetUniformLocation(g_programHandle, "ScreenSize");
    if (screenSizeLoc >= 0)
    {
    glUniform2f(screenSizeLoc, (float)g_winWidth, (float)g_winHeight);
    }
    else
    {
    cout << "ScreenSize"
         << "is not bind to the uniform"
         << endl;
    }
    GLint stepSizeLoc = glGetUniformLocation(g_programHandle, "StepSize");
    
//    GL_ERROR();
    if (stepSizeLoc >= 0)
    {
    glUniform1f(stepSizeLoc, g_stepSize);
    }
    else
    {
    cout << "StepSize"
         << "is not bind to the uniform"
         << endl;
    }

    GLint transferFuncLoc = glGetUniformLocation(g_programHandle, "TransferFunc");
    if (transferFuncLoc >= 0)
    {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, g_tffTexObj);
    glUniform1i(transferFuncLoc, 0);
    }
    else
    {
    cout << "TransferFunc"
         << "is not bind to the uniform"
         << endl;
    }

    GLint backFaceLoc = glGetUniformLocation(g_programHandle, "ExitPoints");
    if (backFaceLoc >= 0)
    {
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, g_bfTexObj);
    glUniform1i(backFaceLoc, 1);
    }
    else
    {
    cout << "ExitPoints"
         << "is not bind to the uniform"
         << endl;
    }
//    GL_ERROR();
    GLint volumeLoc = glGetUniformLocation(g_programHandle, "VolumeTex");
    if (volumeLoc >= 0)
    {
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, g_volTexObj);
    glUniform1i(volumeLoc, 2);
    }
    else
    {
    cout << "VolumeTex"
         << "is not bind to the uniform"
         << endl;
    }
    
}

// init the shader object and shader program
void initShader()
{
// vertex shader object for first pass
    g_bfVertHandle = initShaderObj("/Users/zhoutaotao/Documents/Xcode/vr/vr/shader/backface.vert", GL_VERTEX_SHADER);
// fragment shader object for first pass
    g_bfFragHandle = initShaderObj("/Users/zhoutaotao/Documents/Xcode/vr/vr/shader/backface.frag", GL_FRAGMENT_SHADER);
// vertex shader object for second pass
    g_rcVertHandle = initShaderObj("/Users/zhoutaotao/Documents/Xcode/vr/vr/shader/raycasting.vert", GL_VERTEX_SHADER);
// fragment shader object for second pass
    g_rcFragHandle = initShaderObj("/Users/zhoutaotao/Documents/Xcode/vr/vr/shader/raycasting.frag", GL_FRAGMENT_SHADER);
// create the shader program , use it in an appropriate time
    g_programHandle = createShaderPgm();

}

// link the shader objects using the shader program
void linkShader(GLuint shaderPgm, GLuint newVertHandle, GLuint newFragHandle)
{
    const GLsizei maxCount = 2;
    GLsizei count;
    GLuint shaders[maxCount];
    glGetAttachedShaders(shaderPgm, maxCount, &count, shaders);
     //cout << "get VertHandle: " << shaders[0] << endl;
     //cout << "get FragHandle: " << shaders[1] << endl;
    GL_ERROR();
    for (int i = 0; i < count; i++) {
    glDetachShader(shaderPgm, shaders[i]);
    }
    // Bind index 0 to the shader input variable "VerPos"
    glBindAttribLocation(shaderPgm, 0, "VerPos");
    // Bind index 1 to the shader input variable "VerClr"
    glBindAttribLocation(shaderPgm, 1, "VerClr");
    GL_ERROR();
    glAttachShader(shaderPgm,newVertHandle);
    glAttachShader(shaderPgm,newFragHandle);
    GL_ERROR();
    glLinkProgram(shaderPgm);
    if (GL_FALSE == checkShaderLinkStatus(shaderPgm))
    {
    cerr << "Failed to relink shader program!" << endl;
    exit(EXIT_FAILURE);
    }
    GL_ERROR();
}

// angle of rotation for the camera direction
float angle=0.0;
// actual vector representing the camera's direction
float lx=0.0f,lz=-1.0f;
// XZ position of the camera
float x=0.0f,z=5.0f;

void processSpecialKeys(int key, int xx, int yy) {

    float fraction = 0.1f;

    switch (key) {
        case GLUT_KEY_LEFT :
            angle -= 0.1f;
            lx = sin(angle);
            lz = -cos(angle);
            cout << "l" << endl;
            break;
        case GLUT_KEY_RIGHT :
            angle += 0.1f;
            lx = sin(angle);
            lz = -cos(angle);
            cout << "r" << endl;
            break;
        case GLUT_KEY_UP :
            x += lx * fraction;
            z += lz * fraction;
            cout << "up" << endl;
            break;
        case GLUT_KEY_DOWN :
            x -= lx * fraction;
            z -= lz * fraction;
            cout << "down" << endl;
            break;
    }
    display();
}

void rotateDisplay()
{
    g_angle = (g_angle + 0.1);
    glutPostRedisplay();
}

void display()
{
    glEnable(GL_DEPTH_TEST);
    // render to texture
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_frameBuffer);
    glViewport(0, 0, g_winWidth, g_winHeight);
    linkShader(g_programHandle, g_bfVertHandle, g_bfFragHandle);
    glUseProgram(g_programHandle);
    // cull front face
    render(GL_FRONT);
    glUseProgram(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, g_winWidth, g_winHeight);
    linkShader(g_programHandle, g_rcVertHandle, g_rcFragHandle);

    glUseProgram(g_programHandle);
    rcSetUinforms();

    render(GL_BACK);
    glUseProgram(0);
    
    Mat histImage( bg_h, bg_w, CV_8UC3, Scalar( 255,255,255) );
    hist(histImage);
    drawcurve(histImage, cvred, (float)redChannelWidth, (float)redMean);
    drawcurve(histImage, cvgreen, (float)greenChannelWidth, (float)greenMean);
    drawcurve(histImage, cvblue, (float)blueChannelWidth, (float)blueMean);
    imshow("Intensity Histogram", histImage );
    glutSwapBuffers();
}


// both of the two pass use the "render() function"
// the first pass render the backface of the boundbox
// the second pass render the frontface of the boundbox
// together with the frontface, use the backface as a 2D texture in the second pass
// to calculate the entry point and the exit point of the ray in and out the box.
void render(GLenum cullFace)
{
    glClearColor(0.2f,0.2f,0.2f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //  transform the box
    glm::mat4 projection = glm::perspective(100.0f, (GLfloat)g_winWidth/g_winHeight, 0.1f, 400.f);
    glm::mat4 view = glm::lookAt(glm::vec3(0.5f, 0.5f, 1.5f),
                    glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f));
    //glm::vec3 scale = glm::vec3(0.5f, 0.5f, 0.5f);
    glm::mat4 model = mat4(1.0f);
    //model = glm::scale(model, scale);
    model = glm::rotate(model, glm::radians((float)(-1* stheta)), glm::vec3(1.0f, 0.0f, 0.0f));
    model *= glm::rotate((float)glm::radians(angle), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians((float)( sphi)), glm::vec3(0.0f, 1.0f, 0.0f));
    model *= glm::rotate((float)angle, glm::vec3(0.0f, 1.0f, 0.0f));
    model *= glm::translate(glm::vec3(viewer[0], viewer[1], viewer[2]));
    glm::mat4 mvp = projection * view * model;

    GLuint mvpIdx = glGetUniformLocation(g_programHandle, "MVP");
    
    glUniform3f(glGetUniformLocation(g_programHandle,"ambientColor"),0.2f,0.1f,0.0f);
    glUniform3f(glGetUniformLocation(g_programHandle, "lightPos"), 10.0f,10.0f,5.0f);
    glUniform3f(glGetUniformLocation(g_programHandle, "lightColor"), 1.0f,1.0f,1.0f);
    glUniform3f(glGetUniformLocation(g_programHandle, "cameraPos"),0.5f,0.5f,1.5f);
    
    
    if (mvpIdx >= 0)
    {
        glUniformMatrix4fv(mvpIdx, 1, GL_FALSE, &mvp[0][0]);
    }
    else
    {
        cerr << "can't get the MVP" << endl;
    }
    drawBox(cullFace);
}

void reshape(int w, int h)
{
    g_winWidth = w;
    g_winHeight = h;
    g_texWidth = w;
    g_texHeight = h;
}

void keyboard(unsigned char key, int x, int y)
{

    switch (key)
    {
        case '\x1B':
            exit(EXIT_SUCCESS);
            break;
    }
}

void glui_cb( int control)
{
    redStart = (redMean > redChannelWidth ? redMean - redChannelWidth : 0);
    redEnd = (redMean + redChannelWidth > 256 ? 256 : redMean + redChannelWidth);

    greenStart = (greenMean > greenChannelWidth ? greenMean - greenChannelWidth : 0);
    greenEnd = (greenMean + greenChannelWidth > 256 ? 256 : greenMean + greenChannelWidth);

    blueStart = (blueMean > blueChannelWidth ? blueMean - blueChannelWidth : 0);
    blueEnd = (blueMean + blueChannelWidth > 256 ? 256 : blueMean + blueChannelWidth);
    // operate the command
    if (control == RED_ENABLED_ID)
    {
        if (red_enabled)
        {
            red->enable();
        }
        else
        {
            redSpinner->disable();
            redScrollbar->disable();
            redEnd = redStart - 1;
        }
    }
    else if (control == GREEN_ENABLED_ID)
    {
        if (green_enabled)
        {
            green->enable();
        }
        else
        {
            greenSpinner->disable();
            greenScrollbar->disable();
            greenEnd = greenStart - 1;
        }
    }
    else if (control == BLUE_ENABLED_ID)
    {
        if (blue_enabled)
        {
            blue->enable();
        }
        else
        {
            blueSpinner->disable();
            blueScrollbar->disable();
            blueEnd = blueStart - 1;
        }
    }

    g_tffTexObj = initTFF1DTex();
    glutPostRedisplay();


}

void myGlutIdle(void)
{
    if (glutGetWindow() != main_window)
        glutSetWindow(main_window);

    //glutPostRedisplay
    glutPostRedisplay();
}

int main(int argc, char** argv)
{


    //compute the histgram
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(400, 400);
    main_window = glutCreateWindow(name.data());
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }
  
    glutKeyboardFunc(&keyboard);
    glutSpecialFunc(processSpecialKeys);
    glutDisplayFunc(&display);
    glutIdleFunc(&rotateDisplay);
    glutReshapeFunc(&reshape);
    glutMouseFunc(MouseFunc);
    glutMotionFunc(MotionFunc);

    init(fileName, 256,256,128);
    /***************************GLUI CODE*******************************/
        
        cout << "GLUT VERSOIN = " << GLUI_Master.get_version() << endl;

        GLUI *glui_window = GLUI_Master.create_glui("Transfer Function");
        

        red = new GLUI_Panel(glui_window, "Red");
        green = new GLUI_Panel(glui_window, "Green");
        blue = new GLUI_Panel(glui_window, "Blue");
        opacity = new GLUI_Panel(glui_window, "Opacity");

        // red
        new GLUI_Checkbox(red, "Enabled", &red_enabled, RED_ENABLED_ID, glui_cb);
        redSpinner = new GLUI_Spinner(red, "Width:", &redChannelWidth, -1, glui_cb);
        redSpinner->set_int_limits(0, 127);
        redScrollbar = new GLUI_Scrollbar(red, "Intensity Mean", GLUI_SCROLL_HORIZONTAL, &redMean, -1, glui_cb);
        redScrollbar->set_int_limits(0, 255);
        // green
        new GLUI_Checkbox(green, "Enabled", &green_enabled, GREEN_ENABLED_ID, glui_cb);
        greenSpinner = new GLUI_Spinner(green, "Width:", &greenChannelWidth, -1, glui_cb);
        greenSpinner->set_int_limits(0, 127);
        greenScrollbar = new GLUI_Scrollbar(green, "Intensity Mean", GLUI_SCROLL_HORIZONTAL, &greenMean, -1, glui_cb);
        greenScrollbar->set_int_limits(0, 255);
        // blue
        new GLUI_Checkbox(blue, "Enabled", &blue_enabled, BLUE_ENABLED_ID, glui_cb);
        blueSpinner = new GLUI_Spinner(blue, "Width:", &blueChannelWidth, -1, glui_cb);
        blueSpinner->set_int_limits(0, 127);
        blueScrollbar = new GLUI_Scrollbar(blue, "Intensity Mean", GLUI_SCROLL_HORIZONTAL, &blueMean, -1, glui_cb);
        blueScrollbar->set_int_limits(0, 255);
        // opacity
        (new GLUI_Spinner(opacity, "transparent value:", &opacity_scalar, -1, glui_cb))->set_int_limits(0, 255);
        (new GLUI_Spinner(opacity, "Width:", &opacityWidth, -1, glui_cb))->set_int_limits(0, 127);
        (new GLUI_Scrollbar(opacity, "Intensity Mean", GLUI_SCROLL_HORIZONTAL, &opacityMean, -1, glui_cb))->set_int_limits(0, 255);
        glui_window->set_main_gfx_window( main_window );
        GLUI_Master.set_glutIdleFunc(myGlutIdle);

        /*******************************END GLUI*************************/

    glutMainLoop();
    return EXIT_SUCCESS;
}
