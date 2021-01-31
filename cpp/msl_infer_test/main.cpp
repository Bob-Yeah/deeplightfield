#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "../msl_infer/SynthesisPipeline.h"
#include "../msl_infer/View.h"
#include "../glm/gtx/transform.hpp"

static const struct
{
	float x, y;
	float u, v;
} vertices[4] = {
	{-1.0f, -1.0f, 0.f, 1.f},
	{1.0f, -1.0f, 1.f, 1.f},
	{1.0f, 1.0f, 1.f, 0.f},
	{-1.0f, 1.0f, 0.f, 0.f}};

static const char *vertex_shader_text =
	"#version 300 es\n"
	"uniform mat4 MVP;\n"
	"in vec2 vUV;\n"
	"in vec2 vPos;\n"
	"out vec2 uv;\n"
	"void main()\n"
	"{\n"
	"    gl_Position = MVP * vec4(vPos, 0.0, 1.0);\n"
	"    uv = vUV;\n"
	"}\n";

static const char *fragment_shader_text =
	"#version 300 es\n"
	"#undef lowp\n"
	"#undef mediump\n"
	"#undef highp\n"
	"precision mediump float;\n"
	"out vec4 FragColor;\n"
	"in vec2 uv;\n"
	"uniform sampler2D tex;\n"
	"uniform float R;\n"
	"uniform vec2 foveaCenter;\n"
	"uniform vec2 screenRes;\n"
	"void main()\n"
	"{\n"
	"    if(R<1e-5) {\n"
	"        FragColor = texture(tex, uv);\n"
	"        return;\n"
	"    }\n"
	"    vec2 p = uv * screenRes;\n"
	"    float r = distance(p, foveaCenter);\n"
	"    vec2 coord = (p - foveaCenter) / R / 2.0 + 0.5;\n"
	"    if(coord.x < 0.0 || coord.x > 1.0 || coord.y < 0.0 || coord.y > 1.0) {\n"
	"        FragColor = vec4(0, 0, 0, 0);\n"
	"        return;\n"
	"    }\n"
	"    vec4 c = texture(tex, coord);\n"
	"    float alpha = 1.0 - smoothstep(R * 0.6, R, r);\n"
	"    c.a = c.a * alpha;\n"
	"    FragColor = c;\n"
	"}\n";

void inferFovea(void *o_imageData, View &view)
{
	glm::uvec2 foveaRes(128, 128);
	size_t foveaPixels = foveaRes.x * foveaRes.y;
	size_t totalPixels = foveaPixels;
	size_t samples = 32;

	Camera foveaCam(20, foveaRes / 2u, foveaRes);
	InferPipeline inferPipeline("../nets/fovea_mono/", true, totalPixels, samples);

	auto local_rays = foveaCam.localRays();
	auto rays = sptr<CudaArray<glm::vec3>>(new CudaArray<glm::vec3>(totalPixels));
	auto colors = sptr<CudaArray<glm::vec4>>(new CudaArray<glm::vec4>(totalPixels));

	CudaEvent eStart, eGenRays, eInferred, eEnhanced;

	cudaEventRecord(eStart);

	view.transVectors(rays, local_rays);

	cudaEventRecord(eGenRays);

	inferPipeline.run(colors, rays, view.t(), true);

	cudaEventRecord(eInferred);

	// TODO Enhance

	cudaEventRecord(eEnhanced);

	CHECK_EX(cudaDeviceSynchronize());

	float timeTotal, timeGenRays, timeInfer, timeEnhance;
	cudaEventElapsedTime(&timeTotal, eStart, eEnhanced);
	cudaEventElapsedTime(&timeGenRays, eStart, eGenRays);
	cudaEventElapsedTime(&timeInfer, eGenRays, eInferred);
	cudaEventElapsedTime(&timeEnhance, eInferred, eEnhanced);
	{
		std::ostringstream sout;
		sout << "Fovea => Total: " << timeTotal << "ms (Gen rays: " << timeGenRays
			 << "ms, Infer: " << timeInfer << "ms, Enhance: " << timeEnhance << "ms)";
		Logger::instance.info(sout.str());
	}
	cudaMemcpy(o_imageData, colors->getBuffer(), colors->size(), cudaMemcpyDeviceToHost);
}

void inferOther(void *o_imageData, View &view)
{
	glm::uvec2 midRes(256, 256);
	glm::uvec2 periphRes(230, 256);
	size_t midPixels = midRes.x * midRes.y;
	size_t periphPixels = periphRes.x * periphRes.y;
	size_t totalPixels = midPixels + periphPixels;
	size_t samples = 16;

	Camera midCam(45.0f, {128.0f, 128.0f}, midRes);
	Camera periphCam(110.0f, {115.0f, 128.0f}, periphRes);
	InferPipeline inferPipeline("../nets/periph/", true, totalPixels, samples);

	auto midLocalRays = midCam.localRays();
	auto periphLocalRays = periphCam.localRays();
	auto rays = sptr<CudaArray<glm::vec3>>(new CudaArray<glm::vec3>(totalPixels));
	auto midRays = sptr<CudaArray<glm::vec3>>(new CudaArray<glm::vec3>(*rays, midPixels));
	auto periphRays = sptr<CudaArray<glm::vec3>>(new CudaArray<glm::vec3>((glm::vec3 *)*rays + midPixels, periphPixels));
	auto colors = sptr<CudaArray<glm::vec4>>(new CudaArray<glm::vec4>(totalPixels));

	CudaEvent eStart, eGenRays, eInferred, eEnhanced;

	cudaEventRecord(eStart);

	view.transVectors(midRays, midLocalRays);
	view.transVectors(periphRays, periphLocalRays);

	cudaEventRecord(eGenRays);

	inferPipeline.run(colors, rays, view.t(), true);

	cudaEventRecord(eInferred);

	// TODO Enhance

	cudaEventRecord(eEnhanced);

	CHECK_EX(cudaDeviceSynchronize());

	float timeTotal, timeGenRays, timeInfer, timeEnhance;
	cudaEventElapsedTime(&timeTotal, eStart, eEnhanced);
	cudaEventElapsedTime(&timeGenRays, eStart, eGenRays);
	cudaEventElapsedTime(&timeInfer, eGenRays, eInferred);
	cudaEventElapsedTime(&timeEnhance, eInferred, eEnhanced);
	{
		std::ostringstream sout;
		sout << "Mid & Periph => Total: " << timeTotal << "ms (Gen rays: " << timeGenRays
			 << "ms, Infer: " << timeInfer << "ms, Enhance: " << timeEnhance << "ms)";
		Logger::instance.info(sout.str());
	}
	cudaMemcpy(o_imageData, colors->getBuffer(), colors->size(), cudaMemcpyDeviceToHost);
}

static void error_callback(int error, const char *description)
{
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
}

GLFWwindow *initGl(uint windowWidth, uint windowHeight)
{
	glfwSetErrorCallback(error_callback);
	if (!glfwInit())
		return nullptr;
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	/*glfwWindowHint(GLFW_DEPTH_BITS, 0);
	glfwWindowHint(GLFW_STENCIL_BITS, 0);

	glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	*/
	GLFWwindow *window = glfwCreateWindow(windowWidth, windowHeight, "LearnOpenGL", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return nullptr;
	}
	glfwSetKeyCallback(window, key_callback);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	/*if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }*/

	glewInit();
	glViewport(0, 0, windowWidth, windowHeight);
	glClearColor(0.0f, 0.0f, 0.3f, 1.0f);

	Logger::instance.info("OpenGL is initialized");

	return window;
}

GLuint createGlTexture(uint width, uint height)
{
	GLuint textureID;
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
	return textureID;
}

void checkCompileErrors(unsigned int shader, std::string type)
{
	int success;
	char infoLog[1024];
	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
					  << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
					  << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}

GLuint loadShaderProgram()
{
	GLuint vertex_shader, fragment_shader, program;
	vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
	glCompileShader(vertex_shader);
	checkCompileErrors(vertex_shader, "VERTEX");

	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
	glCompileShader(fragment_shader);
	checkCompileErrors(fragment_shader, "FRAGMENT");

	program = glCreateProgram();
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);
	glLinkProgram(program);
	checkCompileErrors(program, "PROGRAM");

	Logger::instance.info("Shader program is loaded");
	return program;
}

int main(void)
{
	Logger::instance.logLevel = 3;

	GLFWwindow *window;
	GLuint vertex_buffer, program;
	GLint mvp_location, vpos_location, vcol_location;

	window = initGl(800, 800);

	glGenBuffers(1, &vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	program = loadShaderProgram();
	GLuint shaderProp_tex = glGetUniformLocation(program, "tex");
	GLuint shaderProp_R = glGetUniformLocation(program, "R");
	GLuint shaderProp_screenRes = glGetUniformLocation(program, "screenRes");
	GLuint shaderProp_foveaCenter = glGetUniformLocation(program, "foveaCenter");

	mvp_location = glGetUniformLocation(program, "MVP");
	vpos_location = glGetAttribLocation(program, "vPos");
	vcol_location = glGetAttribLocation(program, "vUV");

	glEnableVertexAttribArray(vpos_location);
	glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE,
						  sizeof(vertices[0]), (void *)0);
	glEnableVertexAttribArray(vcol_location);
	glVertexAttribPointer(vcol_location, 2, GL_FLOAT, GL_FALSE,
						  sizeof(vertices[0]), (void *)(sizeof(float) * 2));

	sptr<FoveaSynthesisPipeline> foveaSynthesisPipeline(
		new FoveaSynthesisPipeline({128, 128}, 20, 32));
	sptr<PeriphSynthesisPipeline> periphSynthesisPipeline(
		new PeriphSynthesisPipeline({256, 256}, 45, {230, 256}, 110, 16));
	View view({}, {});
	auto glFoveaTex = foveaSynthesisPipeline->getGlResultTexture(0);
	auto glMidTex = periphSynthesisPipeline->getGlResultTexture(0);
	auto glPeriphTex = periphSynthesisPipeline->getGlResultTexture(1);

	Logger::instance.info("Start main loop");

	auto l = 1.428f;
	glm::vec2 screenRes(1440.0f, 1600.0f);
	glm::mat4 mvp = glm::ortho(-1.f, 1.f, -1.f, 1.f, 1.f, -1.f);

	glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	GLuint queries[1];
	glGenQueries(1, queries);

	while (!glfwWindowShouldClose(window))
	{
		foveaSynthesisPipeline->run(view);
		periphSynthesisPipeline->run(view);

		glClear(GL_COLOR_BUFFER_BIT);

        // Start query 1
        glBeginQuery(GL_TIME_ELAPSED, queries[0]);

		glUseProgram(program);
		glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (float *)&mvp[0][0]);
		glUniform1i(shaderProp_tex, 0);
		glEnable(GL_TEXTURE_2D);
		glActiveTexture(GL_TEXTURE0);

		glUniform1f(shaderProp_R, 0.0f);
		glUniform2f(shaderProp_screenRes, 1440, 1600);
		glUniform2f(shaderProp_foveaCenter, 720, 800);
		glBindTexture(GL_TEXTURE_2D, glPeriphTex);
		glDrawArrays(GL_QUADS, 0, 4);

		glUniform1f(shaderProp_R, screenRes.y * 0.5f * 0.414 / l);
		glUniform2f(shaderProp_screenRes, 1440, 1600);
		glUniform2f(shaderProp_foveaCenter, 720, 800);
		glBindTexture(GL_TEXTURE_2D, glMidTex);
		glDrawArrays(GL_QUADS, 0, 4);

		glUniform1f(shaderProp_R, screenRes.y * 0.5f * 0.176f / l);
		glUniform2f(shaderProp_screenRes, 1440, 1600);
		glUniform2f(shaderProp_foveaCenter, 720, 800);
		glBindTexture(GL_TEXTURE_2D, glFoveaTex);
		glDrawArrays(GL_QUADS, 0, 4);

		glDisable(GL_TEXTURE_2D);

        glEndQuery(GL_TIME_ELAPSED);

        GLint available = 0;
		while (!available)
            glGetQueryObjectiv(queries[0], GL_QUERY_RESULT_AVAILABLE, &available);
        // timer queries can contain more than 32 bits of data, so always
        // query them using the 64 bit types to avoid overflow
        GLuint64 timeElapsed = 0;
		glGetQueryObjectui64v(queries[0], GL_QUERY_RESULT, &timeElapsed);

		{
			std::ostringstream sout;
			sout << "Blending: " << timeElapsed / 10000 / 100.0f << "ms" << std::endl;
			Logger::instance.info(sout.str());
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	foveaSynthesisPipeline = nullptr;
	periphSynthesisPipeline = nullptr;

	glfwDestroyWindow(window);

	glfwTerminate();
	exit(EXIT_SUCCESS);
}
