#version 400

in vec3 deColor;
in vec3 normals;
in vec3 fragPosition;
in vec2 daUV;
in mat3 where;
in vec3 tangent;

uniform sampler2D textured;
uniform vec3 cameraPosition;
uniform vec3 light;
uniform bool diffuseOn;
uniform bool specularOn;

out vec3 FinalColor;

void main()
{
	vec3 min = vec3(0.0f, 0.0f, 0.0f);
	vec3 max = vec3(1.0f, 1.0f, 1.0f);

	//Rotate the Normals
	vec3 newTangent = where * tangent;
	vec3 bitangent = cross(newTangent, normals);
	mat3 normalRotationMatrix = mat3(newTangent, bitangent, normals);
	vec3 normalMapPreRotation = normalize(2 * vec3(texture(textured, daUV)) - vec3(1.0f));
	vec3 normalMap = normalize(normalRotationMatrix * normalMapPreRotation);

	//Get the Diffuse Lighting
	vec3 normalized = normalize(light - fragPosition);
	float diffuse = dot(normalized, normalMap);
	vec3 totalDiffuse = diffuse * vec3(1.0f, 1.0f, 1.0f);
	totalDiffuse = clamp(totalDiffuse, min, max);

	//Ambient Lighting
	float ambientColor = 0.15;
	vec3 ambient = vec3(ambientColor, ambientColor, ambientColor);
	vec3 lightingColor = clamp(totalDiffuse + ambient, min, max);
	
	vec3 postDiffuseLight = vec3(0.5f, 0.5f, 0.5f);//vec3(texture(textured, daUV));

	if(diffuseOn){
		postDiffuseLight = lightingColor * postDiffuseLight;
	}

	vec3 postSpecularLight = vec3(0.0f);
	if(specularOn){
		vec3 specularReflect = reflect(normalized * -1, normalMap);
		vec3 eyePosition = normalize(cameraPosition - fragPosition);
		float specularDotProduct = pow(dot(eyePosition, specularReflect), 150 * 0.25);
		postSpecularLight = clamp(normalize(vec3(0.1f, 0.1f, 1.0f)) * specularDotProduct, min, max);
	}
	FinalColor = postDiffuseLight + postSpecularLight;
}
