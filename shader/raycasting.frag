#version 400

in vec3 EntryPoint;
in vec4 ExitPointCoord;

uniform sampler2D ExitPoints;
uniform sampler3D VolumeTex;
uniform sampler1D TransferFunc;  
uniform float     StepSize;
uniform vec2      ScreenSize;

uniform vec3 ambientColor;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 cameraPos;

layout (location = 0) out vec4 FragColor;

void main()
{
    // ExitPointCoord
    vec3 exitPoint = texture(ExitPoints, gl_FragCoord.st/ScreenSize).xyz;
    // that will actually give you clip-space coordinates rather than
    // normalised device coordinates, since you're not performing the perspective
    // division which happens during the rasterisation process (between the vertex
    // shader and fragment shader
    // vec2 exitFragCoord = (ExitPointCoord.xy / ExitPointCoord.w + 1.0)/2.0;
    // vec3 exitPoint  = texture(ExitPoints, exitFragCoord).xyz;
    if (EntryPoint == exitPoint)
    	//background need no raycasting
    	discard;
    vec3 dir = exitPoint - EntryPoint;
    float len = length(dir); // the length from front to back is calculated and used to terminate the ray
    vec3 deltaDir = normalize(dir) * StepSize;
    float deltaDirLen = length(deltaDir);
    vec3 voxelCoord = EntryPoint;
    vec4 colorAcum = vec4(0.0); // The dest color
    float alphaAcum = 0.0;                // The  dest alpha for blending
    float intensity;
    float lengthAcum = 0.0;
    vec4 colorSample; // The src color 
    float alphaSample; // The src alpha
    // backgroundColor
    vec4 bgColor = vec4(1.0, 1.0, 1.0, 0.0);
 
    for(int i = 0; i < 1600; i++)
    {
    	intensity =  texture(VolumeTex, voxelCoord).x;
    	colorSample = texture(TransferFunc, intensity);
    	// modulate the value of colorSample.a
    	// front-to-back integration
    	if (colorSample.a > 0.0) {
    	    // accomodate for variable sampling rates (base interval defined by mod_compositing.frag)
    	    colorSample.a = 1.0 - pow(1.0 - colorSample.a, StepSize*200.0f);
    	    colorAcum.rgb += (1.0 - colorAcum.a) * colorSample.rgb * colorSample.a;
    	    colorAcum.a += (1.0 - colorAcum.a) * colorSample.a;
    	}
    	voxelCoord += deltaDir;
    	lengthAcum += deltaDirLen;
    	if (lengthAcum >= len )
    	{	
    	    colorAcum.rgb = colorAcum.rgb*colorAcum.a + (1 - colorAcum.a)*bgColor.rgb;		
    	    break;  // terminate if opacity > 1 or the ray is outside the volume	
    	}	
    	else if (colorAcum.a > 1.0)
    	{
    	    colorAcum.a = 1.0;
    	    break;
    	}
    }
    FragColor = colorAcum;
}
