(function(f,k,A,r){var p=function(){function c(b){if(b&&g.w(b)){var e={name:null,B:null,f:null};if(0<=b.indexOf(".")){b=b.split(".",2);var c=b[0];b=b[1];if(!d[c])return;e.B=c}if(0<=b.indexOf(":")){b=b.split(":",2);c=b[0];b=b[1];if(!a[c])return;e.f=c}if(e.f||x.prototype[b])return e.name=b,e}}var a={},d={},b={create:function(a,b,c){if(!g.h(a))return!1;b=b||a;if(d[b])return!1;d[b]=new x(a,b,c);return!0},exec:function(b){var e=Array.prototype.slice.call(arguments,1);if("exec"===b)return!1;if(p[b])return p[b].apply(null,
e);var n=c(b);if(!n)return!1;var l=a[n.f];if(g.isArray(l))return a[n.f].push(Array.prototype.slice.call(arguments,0)),!0;if(l&&!l[n.name])return!1;for(var f=n.B?[d[n.B]]:g.values(d),y=e.length&&g.u(e[-1])?e[-1].callback:r,z=g.J(),t=!0,k=0;k<f.length;k++){var m=f[k];y&&(e[-1].callback=z.D());t=(l||m)[n.name].apply(m,e)&&t}z.complete(y);return t},get:function(b,a){a=a||g.c;"identity"===b&&u.P(m.b(function(b){a(b)}))},register:function(b,d){if(!a[b]||!g.o(d))return!1;var c=a[b];a[b]=d();setTimeout(function(){for(var b=
0;b<c.length;b++)p.exec.apply(null,c[b])});return!0},require:function(b){if(a[b])return!1;a[b]=[];v.ga("/plugins/"+b+".js");return!0},set:function(a,d){b.i=b.i||function(b,a){p.exec("associate",b,{callback:a})};switch(a){case "captureEmailForms":d?w.F(b.i):w.ea(b.i)}}};return b}(),w=function(){function c(){n=function(){k.addEventListener("submit",h,!1);setTimeout(function(){k.removeEventListener("submit",h,!1)});return!0};k.addEventListener("submit",n,!0);k.addEventListener("click",b,!0)}function a(b){var a=
/\S+@\S+.\S/;b=b.getElementsByTagName("input");for(var d,c=0;c<b.length;c++)if(d=b[c],!d.hasAttribute("center-ignore")&&(d.hasAttribute("center-associate-email")||"email"===d.type||0<=d.name.toLowerCase().replace(/[^a-z]+/gi,"").indexOf("email")||0<=d.id.toLowerCase().replace(/[^a-z]+/gi,"").indexOf("email")||0<=d.className.toLowerCase().replace(/[^a-z]+/gi,"").indexOf("email")))return a.test(d.value)?d.value:null;return null}function d(b,d){function c(){if(!b.I){b.I=!0;if(d){var a=k.createElement("input");
a.type="hidden";for(var e=["name","value"],h=0;h<e.length;h++){var g=a,l=e[h],f=void 0,f=f||l,l=d.getAttribute(l);null!==l&&g.setAttribute(f,l)}b.appendChild(a);a=["action","enctype","method","novalidate","target"];for(e=0;e<a.length;e++)h=b,f="form"+a[e],g=(g=a[e])||f,f=d.getAttribute(f),null!==f&&h.setAttribute(g,f)}HTMLFormElement.prototype.centerOldSubmit.call(b)}}try{for(var e=a(b),h=g.J(),l=0;l<m.length;l++)m[l].call(null,e,h.D());h.complete(c)}catch(f){c()}setTimeout(c,2E3)}function b(b){var a=
b.target;if(a&&0<="INPUT BUTTON".indexOf(a.tagName)&&0<="submit image button".indexOf(a.type)){b=b.target;a=b;if(a.form)a=a.form;else for(;a&&"FORM"!==a.tagName;)a=a.parentElement;a&&!a.hasAttribute("center-ignore")&&(a.centerActor=b)}}function h(b){if(!1!==b.returnValue&&!b.defaultPrevented){var c=b.target;if(c&&!c.hasAttribute("center-ignore")&&a(c))return d(c,c.centerActor),b.preventDefault(),!1}}function e(){HTMLFormElement.prototype.centerOldSubmit=HTMLFormElement.prototype.submit;HTMLFormElement.prototype.submit=
function(){if(this.hasAttribute("center-ignore")||!a(this))return HTMLFormElement.prototype.centerOldSubmit.call(this);d(this,null)}}var n,l=!1,m=[];return{F:function(b){!l&&k.addEventListener&&f.HTMLFormElement&&!l&&(c(),e(),l=!0);m.push(b)},ea:function(a){a=m.indexOf(a);-1<a&&m.splice(a,1);!m.length&&l&&l&&(k.removeEventListener("submit",n,!0),k.removeEventListener("click",b,!0),HTMLFormElement.prototype.submit=void 0,l=!1)}}}(),v=function(){return{H:function(c,a,d){d=d||g.c;var b={method:"GET"};
b.url="https://api.leadpages.io"+c+"?"+g.S(a);var h=new XMLHttpRequest;"withCredentials"in h?(h.open(b.method,b.url,!0),h.onreadystatechange=function(){4===h.readyState&&(b.code=h.U,400<=h.status?d(h.responseText,b):d(null,b))},h.ontimeout=function(){b.U=0;d("TIMEOUT",b)},h.send()):(c=new Image,g.G(c,"load",function(){d(null,b)}),c.src=b.url)},ga:function(c){var a=k.createElement("script");a.async=!0;a.src="https://js.center.io"+c;k.getElementsByTagName("head")[0].appendChild(a)}}}(),m=function(){var c=
[],a={flush:function(){if(c.length){for(var a=[],b=[],h=[],e=0;e<c.length;e++)a.push(c[e].kind),b.push(c[e].label),h.push(c[e].value);c.length=0;a={version:"1.6.2",correlateBy:g.M("center-correlate-by")||g.N(),origin:"center-js",kind:a,label:b,value:h};v.H("/analytics/v1/observations/capture",a)}},g:function(d,b,h){c.push({kind:d,label:b,value:h});a.C=a.C||g.V(a.flush,4E3,!1);a.C()},A:function(d,b){if(f.performance&&f.performance.getEntriesByName){var c=performance.getEntriesByName(b)[0];c&&c.duration&&
a.g("timer",d,c.duration)}},da:function(d){if(!(f.performance&&f.performance.mark&&f.performance.measure&&f.performance.getEntriesByName))return g.c;var b="center-"+d+"-s",c="center-"+d+"-e",e="center-"+d;performance.mark(b);return function(){performance.mark(c);performance.measure(e,b,c);a.A(d,e)}},R:function(d){if(d)try{var b={uid:g.j("centerVisitorId")||"",url:q.O(),ua:navigator.userAgent,n:(d.name||"").substr(0,100),m:(d.message||"").substr(0,100),s:(d.stack||"").replace(/\s+/gi," ").substr(0,
500)};a.g("text","error",g.S(b));a.flush()}catch(c){}},b:function(d,b){b=b||f;return function(){try{return d.apply(b,arguments)}catch(c){a.R(c)}}},ha:function(d){for(var b in d)d.hasOwnProperty(b)&&g.o(d[b])&&(d[b]=a.b(d[b],d))}};return a}(),q=function(){return{W:function(){return{x:f.innerWidth||(k.documentElement||{}).clientWidth||(k.body||{}).clientWidth||0,y:f.innerHeight||(k.documentElement||{}).clientHeight||(k.body||{}).clientHeight||0}},X:function(){return k.referrer},Y:function(c){function a(a){a=
Math.abs(Math.floor(a));return(10>a?"0":"")+a}c=c||new Date;c=-c.getTimezoneOffset();return(0<=c?"+":"-")+a(c/60)+":"+a(c%60)},O:function(){return f.location.toString().replace(/ /g,"%20")},Z:function(){var c=g.L(q.$());(c=c.utm_email||c.center_email||r)&&(c=c.replace(/ /g,"+"));return c},$:function(){return f.location.search||""}}}(),x=function(){function c(d,b,c){var e=this;e.ca=d;e.name=b;e.options=g.K(c||{},a);e.options.captureEmailForms&&w.F(function(b,a){e.associate(b,{callback:a})});e.options.captureEmailURLs&&
(d=q.Z())&&e.associate(d)}var a={captureEmailForms:!1,captureEmailURLs:!0,customId:""};c.prototype.associate=function(a,b){b=b||{};b.email=a;return this.send("association",null,null,null,b)};c.prototype.send=function(a,b,c,e,f){var l=this,k={};k.callback=g.c;k.customId="";f=g.K(f||{},l.options,k);u.P(m.b(function(g){var k=q.W();g={k:a||"",a:b||"",l:c||"",v:e||"",e:f.email||"",pid:l.ca,uid:g.uid,sid:g.sid,cid:f.customId,uri:q.O().substr(0,1E3),rf:q.X().substr(0,500),rx:k.x,ry:k.y,tz:q.Y()};v.H("/analytics/v1/events/capture",
g,function(a,b){m.A("send-events",b.url);a?(m.R({name:"HTTPError",message:b.method+" "+b.url+" "+b.U,stack:a}),m.flush(),f.callback(!1)):f.callback()})}));return!0};return c}(),g=function(){var c={G:function(a,d,b,c){a.addEventListener?a.addEventListener(d,b,c):a.attachEvent&&a.attachEvent("on"+d,b)},J:function(){return new function(){function a(){e>=c&&d&&!b&&(b=!0,d.call(f))}var d=null,b=!1,c=0,e=0;return{D:function(){c++;return function(){e++;setTimeout(a)}},complete:function(a){d=a}}}},V:function(a,
d,b){function c(){var g=this,f=Array.prototype.slice.call(arguments,0),h=b&&!e;clearTimeout(e);e=setTimeout(function(){e=null;b||a.apply(g,f)},d);h&&a.apply(g,f)}var e;c.clear=function(){clearTimeout(e);e=null};return c},K:function(a){var d=Array.prototype.slice.call(arguments,1);if(!d.length)return a;for(var b=0;b<d.length;b++)for(var c=d[b],e=g.keys(c),f=0;f<e.length;f++){var l=e[f];a[l]===r&&(a[l]=c[l])}return a},L:function(a){if(!g.w(a))return null;if(!a)return{};var d={};a=a.replace(/(^\?)/,
"").split("&");for(var b=0;b<a.length;b++){var c=a[b].split("=");if(c[1]&&0<=c[1].indexOf(","))try{d[decodeURIComponent(c[0])]=decodeURIComponent(c[1]).split(",")}catch(e){}else try{d[decodeURIComponent(c[0])]=decodeURIComponent(c[1])}catch(e){}}return d},I:function(a){return decodeURIComponent(k.cookie.replace(new RegExp("(?:(?:^|.*;)\\s*"+encodeURIComponent(a).replace(/[\-\.\+\*]/g,"\\$&")+"\\s*\\=\\s*([^;]*).*$)|^.*$"),"$1"))||null},M:function(a,d){d=d||f.location.search;var b=(new RegExp("[?&]"+
encodeURIComponent(a)+"(=([^&#]*)|&|#|$)")).exec(d);return b?b[2]?decodeURIComponent(b[2].replace(/\+/g," ")):"":null},ia:function(a){try{return localStorage.getItem(a)}catch(d){return null}},j:function(a){try{return sessionStorage.getItem(a)}catch(d){return null}},N:function(){return"xxxxxxxxxxxxxxxxxxxxxx".replace(/[x]/g,function(){return"23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz".charAt(Math.floor(57*Math.random()))})},isArray:function(a){return!(!a||!a.constructor||a.constructor!==
Array)},o:function(a){return!(!a||"function"!==typeof a)},u:function(a){return!(!a||"object"!==typeof a)},w:function(a){return"string"===typeof a},keys:function(a){if(!g.u(a))return[];var d=[],b;for(b in a)a.hasOwnProperty(b)&&d.push(b);return d},ja:function(a){return Array.prototype.slice.call(a,0)},c:function(){},fa:function(a,d,b,c){a.removeEventListener?a.removeEventListener(d,b,c):a.detachEvent&&a.detachEvent("on"+d,b)},S:function(a){var d="",b;for(b in a)a.hasOwnProperty(b)&&(d+="&"+encodeURIComponent(b)+
"=",d=c.isArray(a[b])?d+encodeURIComponent(a[b]).replace(/%2C/g,","):d+encodeURIComponent(a[b]));return d.substr(1,d.length)},ka:function(a,d,b,c,e){k.cookie=encodeURIComponent(a)+"="+encodeURIComponent(d)+"; expires=Fri, 31 Dec 9999 23:59:59 GMT"+(b?"; domain="+b:"")+(c?"; path="+c:"")+(e?"; secure":"")},la:function(a,d){try{return localStorage.setItem(a,d)}catch(b){return null}},T:function(a,d){try{return sessionStorage.setItem(a,d)}catch(b){return null}},h:function(a){return c.w(a)?/^[23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz]{22}$/.test(a):
!1},values:function(a){if(!g.u(a))return[];var d=[],b;for(b in a)a.hasOwnProperty(b)&&d.push(a[b]);return d}};return c}(),u=function(){function c(){for(var c=0;c<b.length;c++)b[c].call(f,{uid:d,sid:a});b.length=0}var a=null,d=null,b=[],h={aa:function(a){var b=m.da("load-identify"),c=k.createElement("iframe"),d=m.b(function(h){0==="https://js.center.io".indexOf(h.origin||(h.originalEvent||{}).origin)&&(h=g.L(h.data))&&h.s&&g.h(h.id)&&(b(),g.fa(f,"message",d,!1),c.parentNode&&c.parentNode.removeChild(c),
a(h))});g.G(f,"message",d,!1);c.src="https://js.center.io/identify.html";c.style.display="none";c.style.visibility="hidden";c.style.position="absolute";c.style.left="-9999px";c.style.top="-9999px";k.getElementsByTagName("head")[0].appendChild(c)},ba:function(){a=g.j("centerSessionId");g.h(a)||(a=g.N(),g.T("centerSessionId",a));d=g.j("centerVisitorId");g.h(d)?(m.g("counter","ident-cache","1"),c()):h.aa(m.b(function(a){m.g("counter",a.s,"1");d=a.id;g.T("centerVisitorId",d);c()}))},P:function(c){d?c.call(f,
{uid:d,sid:a}):b.push(c)}};return h}();m.ha(p);m.b(function(){if(null===g.M("center-no-load")){var c=f[A]||"center";f[c]=f[c]||g.c;var a=f[c].q||[];f[c]=function(a){return g.o(a)?(a(f[c]),!0):p.exec.apply(p,arguments)};f[c].hash="1cab131500a78ee82900b2dd9bc7bca1313ad593";f[c].version="1.6.2";f[c].loaded=!0;u.ba();for(var d=0;d<a.length;d++)f[c].apply(f[name],a[d]);m.A("load-center","https://js.center.io/center.js")}})()})(window,document,"LeadPagesCenterObject");
