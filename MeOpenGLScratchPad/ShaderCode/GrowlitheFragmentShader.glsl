#version 400

in vec3 deColor;
in vec2 daUV;
in mat3 where;
in vec3 fragPosition;
in vec3 normals;

uniform sampler2D textured;
uniform vec3 cameraPosition;
uniform vec3 light;
uniform vec3 dominateColor;

out vec3 FinalColor;


void main()
{
	vec3 min = vec3(0.0f, 0.0f, 0.0f);
	vec3 max = vec3(1.0f, 1.0f, 1.0f);

	//Get the Diffuse Lighting
	vec3 normalized = normalize(light - fragPosition);
	float diffuse = dot(normalized, normals);
	vec3 totalDiffuse = diffuse * vec3(1.0f, 1.0f, 1.0f);
	totalDiffuse = clamp(totalDiffuse, min, max);

	//Ambient Lighting
	float ambientColor = 0.95;
	vec3 ambient = vec3(ambientColor, ambientColor, ambientColor);
	vec3 lightingColor = clamp(totalDiffuse + ambient, min, max);
	
	vec3 postDiffuseLight = vec3(0.5f, 0.5f, 0.5f);//vec3(texture(textured, daUV));
	postDiffuseLight = lightingColor * postDiffuseLight;

	vec3 postSpecularLight = vec3(0.0f);
	vec3 specularReflect = reflect(normalized * -1, normals);
	vec3 eyePosition = normalize(cameraPosition - fragPosition);
	float specularDotProduct = pow(dot(eyePosition, specularReflect), 150 * 0.25);
	postSpecularLight = clamp(normalize(vec3(1.0f, 0.1f, 0.1f)) * specularDotProduct, min, max);

	FinalColor = vec3(0.75f, 0.25f, 0.25f);//deColor * ( postDiffuseLight + postSpecularLight);
}